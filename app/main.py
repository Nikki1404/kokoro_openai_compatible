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


# =====================================================
# CONFIG
# =====================================================
CONFIG_PATH = os.getenv("CONFIG_PATH", "app/config.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
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

KOKORO_REPO_ID = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")

app = FastAPI()
pipeline: Optional[KPipeline] = None


# =====================================================
# HELPERS
# =====================================================
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
    chunks: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        w = s.split()
        if len(w) <= max_words:
            chunks.append(s)
        else:
            for i in range(0, len(w), max_words):
                chunks.append(" ".join(w[i : i + max_words]))
    return chunks


# =====================================================
# OpenAI-compatible schema
# =====================================================
class OpenAISpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: Optional[str] = None
    response_format: Optional[Literal["mp3", "pcm"]] = "mp3"
    speed: Optional[float] = None


# =====================================================
# STARTUP
# =====================================================
@app.on_event("startup")
async def startup():
    global pipeline

    device = get_device()

    logger.info("========================================")
    logger.info("Starting Kokoro TTS Service")
    logger.info(f"Device: {device}")
    logger.info(f"Repo ID: {KOKORO_REPO_ID}")

    if torch is not None:
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available. Running on CPU.")

    try:
        pipeline = KPipeline(
            repo_id=KOKORO_REPO_ID,
            lang_code=DEFAULT_LANG,
            device=device
        )
        logger.info("Kokoro pipeline initialized successfully.")
        logger.info("========================================")

    except Exception as e:
        logger.error("Kokoro initialization failed:")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Kokoro initialization failed: {e}")


@app.get("/healthz")
def healthz():
    return {"ok": True}


# =====================================================
# STREAM GENERATORS
# =====================================================
async def kokoro_pcm_stream(text: str, voice: str, speed: float) -> AsyncGenerator[bytes, None]:

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    chunks = chunk_text(text, WORD_THRESHOLD) if CHUNK_ENABLED else [text]

    total_audio_samples = 0
    first_chunk_time = None
    generation_start = time.perf_counter()

    logger.info(f"TTS generation started | chunks={len(chunks)}")

    for chunk in chunks:
        gen = pipeline(chunk, voice=voice, speed=speed, split_pattern=SPLIT_PATTERN)

        for (_gs, _ps, audio) in gen:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
                ttfa_ms = (first_chunk_time - generation_start) * 1000
                logger.info(f"TTFA (server-side): {ttfa_ms:.2f} ms")

            a = as_numpy_float(audio)
            total_audio_samples += len(a)

            yield float_to_s16le_bytes(a)
            await asyncio.sleep(0)

    total_audio_sec = total_audio_samples / SR
    total_gen_ms = (time.perf_counter() - generation_start) * 1000

    logger.info(
        f"TTS generation completed | "
        f"audio_duration={total_audio_sec:.2f}s | "
        f"generation_time={total_gen_ms:.2f}ms"
    )


# =====================================================
# ENDPOINT
# =====================================================
@app.post("/v1/audio/speech")
async def audio_speech(req: OpenAISpeechRequest, request: Request):

    request_start = time.perf_counter()

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` is empty")

    voice = (req.voice or DEFAULT_VOICE).strip()
    speed = float(req.speed if req.speed is not None else DEFAULT_SPEED)
    fmt = (req.response_format or "mp3").lower()

    logger.info("--------------------------------------------------")
    logger.info("New TTS Request Received")
    logger.info(f"Text Length: {len(text)} chars | Words: {len(text.split())}")
    logger.info(f"Voice: {voice} | Speed: {speed} | Format: {fmt}")

    # Log first 200 chars only to avoid flooding logs
    logger.info(f"Input Text (preview): {text[:200]}{'...' if len(text) > 200 else ''}")

    pcm_gen = kokoro_pcm_stream(text=text, voice=voice, speed=speed)

    if fmt in ("pcm", "pcm16", "s16", "raw"):
        response = StreamingResponse(pcm_gen, media_type="application/octet-stream")
    elif fmt == "mp3":
        mp3_gen = mp3_stream_via_ffmpeg(pcm_gen, SR, MP3_BITRATE)
        response = StreamingResponse(mp3_gen, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")

    total_req_ms = (time.perf_counter() - request_start) * 1000
    logger.info(f"Request accepted | setup_time={total_req_ms:.2f}ms")
    logger.info("--------------------------------------------------")

    return response


# =====================================================
# MP3 STREAMER (UNCHANGED)
# =====================================================
async def mp3_stream_via_ffmpeg(
    pcm_gen: AsyncGenerator[bytes, None],
    sr: int,
    bitrate: str,
) -> AsyncGenerator[bytes, None]:

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

    async def feed_pcm():
        async for chunk in pcm_gen:
            proc.stdin.write(chunk)
            await proc.stdin.drain()
        proc.stdin.close()

    asyncio.create_task(feed_pcm())

    while True:
        out = await proc.stdout.read(4096)
        if not out:
            break
        yield out
        await asyncio.sleep(0)
