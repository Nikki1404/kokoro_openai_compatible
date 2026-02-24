#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import os
import re
import time
import traceback
from typing import AsyncGenerator, Optional, Literal

import numpy as np
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

try:
    import torch
except Exception:
    torch = None

from kokoro import KPipeline


# =========================================================
# Config + Logging Setup
# =========================================================

CONFIG_PATH = os.getenv("CONFIG_PATH", "app/config.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("kokoro-openai-streaming")

SR = int(CONFIG.get("sample_rate", 24000))
DEFAULT_SPEED = float(CONFIG.get("speed", 1.0))
DEFAULT_VOICE = str(CONFIG.get("voice", "af_heart"))
DEFAULT_LANG = str(CONFIG.get("lang_code", "a"))

SPLIT_PATTERN = CONFIG.get("pipeline", {}).get("split_pattern", r"\n+")
CHUNK_ENABLED = bool(CONFIG.get("chunking", {}).get("enabled", True))
WORD_THRESHOLD = int(CONFIG.get("chunking", {}).get("word_threshold", 12))
MP3_BITRATE = str(CONFIG.get("mp3", {}).get("bitrate", "48k"))

app = FastAPI()
pipeline: Optional[KPipeline] = None


# =========================================================
# Helpers
# =========================================================

def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def as_numpy_float(audio) -> np.ndarray:
    if torch is not None and isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(a, -1.0, 1.0)


def float_to_s16le_bytes(a: np.ndarray) -> bytes:
    pcm16 = (a * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def chunk_text(text: str, max_words: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        w = s.split()
        if len(w) <= max_words:
            chunks.append(s)
        else:
            for i in range(0, len(w), max_words):
                chunks.append(" ".join(w[i:i + max_words]))

    return chunks


def media_type_for(fmt: str) -> str:
    if fmt == "mp3":
        return "audio/mpeg"
    if fmt in ("pcm", "pcm16", "s16", "raw"):
        return "application/octet-stream"
    return "application/octet-stream"


# =========================================================
# OpenAI-Compatible Schema
# =========================================================

class OpenAISpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: Optional[str] = None
    response_format: Optional[Literal["mp3", "pcm"]] = "mp3"
    speed: Optional[float] = None


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
async def startup():
    global pipeline

    device = get_device()
    logger.info("Initializing Kokoro pipeline...")
    logger.info(f"Selected device: {device}")

    pipeline = KPipeline(
        lang_code=DEFAULT_LANG,
        device=device,
        repo_id="hexgrad/Kokoro-82M"
    )

    gpu = None
    if torch is not None and torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda"

    logger.info(
        f"Kokoro initialized | device={device} | gpu={gpu} | "
        f"lang={DEFAULT_LANG} | sample_rate={SR}"
    )


# =========================================================
# Health & Info
# =========================================================

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1")
def v1_root():
    return {"ok": True, "service": "kokoro-openai-streaming", "sr": SR}


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail}},
    )


# =========================================================
# PCM Streaming Generator (Lowest Latency)
# =========================================================

async def kokoro_pcm_stream(text: str, voice: str, speed: float) -> AsyncGenerator[bytes, None]:

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    chunks = chunk_text(text, WORD_THRESHOLD) if CHUNK_ENABLED else [text]

    logger.info(f"Text split into {len(chunks)} chunk(s)")

    first_audio = True
    total_samples = 0

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} | words={len(chunk.split())}")

        gen = pipeline(chunk, voice=voice, speed=speed, split_pattern=SPLIT_PATTERN)

        for (_gs, _ps, audio) in gen:
            a = as_numpy_float(audio)
            total_samples += len(a)

            if first_audio:
                logger.info("First audio segment generated (TTFA reached)")
                first_audio = False

            yield float_to_s16le_bytes(a)
            await asyncio.sleep(0)

    total_audio_sec = total_samples / SR
    logger.info(f"Total audio generated: {total_audio_sec:.2f} seconds")


# =========================================================
# MP3 Streaming via FFmpeg (True Streaming)
# =========================================================

async def mp3_stream_via_ffmpeg(
    pcm_gen: AsyncGenerator[bytes, None],
    sr: int,
    bitrate: str,
) -> AsyncGenerator[bytes, None]:

    logger.info("Starting FFmpeg process for MP3 encoding")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sr),
        "-i", "pipe:0",
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", bitrate,
        "-f", "mp3",
        "-flush_packets", "1",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    first_mp3 = True

    async def feed_pcm():
        try:
            async for chunk in pcm_gen:
                proc.stdin.write(chunk)
                await proc.stdin.drain()
            proc.stdin.close()
        except Exception:
            try:
                proc.stdin.close()
            except Exception:
                pass

    feeder = asyncio.create_task(feed_pcm())

    try:
        while True:
            out = await proc.stdout.read(4096)
            if out:
                if first_mp3:
                    logger.info("First MP3 chunk emitted")
                    first_mp3 = False
                yield out
            else:
                break
            await asyncio.sleep(0)
    finally:
        await feeder
        rc = await proc.wait()
        logger.info(f"FFmpeg exited with return code {rc}")


# =========================================================
# OpenAI-Compatible Endpoint
# =========================================================

@app.post("/v1/audio/speech")
async def audio_speech(req: OpenAISpeechRequest, request: Request):

    client_ip = request.client.host if request.client else "unknown"

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` is empty")

    voice = (req.voice or DEFAULT_VOICE).strip()
    speed = float(req.speed if req.speed is not None else DEFAULT_SPEED)
    fmt = (req.response_format or "mp3").lower()

    logger.info("========== NEW REQUEST ==========")
    logger.info(f"Client IP: {client_ip}")
    logger.info(f"Model: {req.model}")
    logger.info(f"Voice: {voice}")
    logger.info(f"Speed: {speed}")
    logger.info(f"Format: {fmt}")
    logger.info(f"Input length: {len(text)} chars")
    logger.info(f"Input text: {text[:500]}{'...' if len(text) > 500 else ''}")

    start = time.perf_counter()

    try:
        pcm_gen = kokoro_pcm_stream(text=text, voice=voice, speed=speed)

        if fmt in ("pcm", "pcm16", "s16", "raw"):
            logger.info("Streaming PCM (lowest latency mode)")
            return StreamingResponse(pcm_gen, media_type=media_type_for(fmt))

        if fmt == "mp3":
            logger.info(f"Streaming MP3 | bitrate={MP3_BITRATE}")
            mp3_gen = mp3_stream_via_ffmpeg(pcm_gen=pcm_gen, sr=SR, bitrate=MP3_BITRATE)
            return StreamingResponse(mp3_gen, media_type=media_type_for(fmt))

        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")

    except asyncio.CancelledError:
        logger.warning("Client disconnected during streaming.")
        raise

    except Exception:
        logger.error("Error during TTS processing")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        total_ms = (time.perf_counter() - start) * 1000.0
        logger.info(f"Request completed | total_handler_time={total_ms:.1f} ms")
        logger.info("=================================")
