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


# -----------------------------
# Config + logging
# -----------------------------
CONFIG_PATH = os.getenv("CONFIG_PATH", "app/config.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
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


# -----------------------------
# Helpers
# -----------------------------
def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def as_numpy_float(audio) -> np.ndarray:
    """Return float32 mono PCM [-1..1]."""
    if torch is not None and isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(a, -1.0, 1.0)


def float_to_s16le_bytes(a: np.ndarray) -> bytes:
    """float32 [-1..1] -> int16 little-endian PCM bytes"""
    pcm16 = (a * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def chunk_text(text: str, max_words: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text]

    # sentence-first chunking, then word chunking
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


def media_type_for(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt == "mp3":
        return "audio/mpeg"
    if fmt in ("pcm", "pcm16", "s16", "raw"):
        return "application/octet-stream"
    if fmt == "wav":
        return "audio/wav"
    return "application/octet-stream"


# -----------------------------
# OpenAI-compatible schema
# -----------------------------
class OpenAISpeechRequest(BaseModel):
    model: str = "kokoro"  # ignored but required by SDK
    input: str
    voice: Optional[str] = None
    response_format: Optional[Literal["mp3", "pcm"]] = "mp3"
    speed: Optional[float] = None


@app.on_event("startup")
async def startup():
    global pipeline
    device = get_device()
    pipeline = KPipeline(lang_code=DEFAULT_LANG, device=device)

    gpu = None
    if torch is not None and torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda"

    logger.info(f"Kokoro initialized (device={device}, gpu={gpu}, lang={DEFAULT_LANG}, sr={SR})")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1")
def v1_root():
    return {"ok": True, "service": "kokoro-openai-streaming", "sr": SR}


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    # OpenAI SDK expects JSON errors sometimes; this keeps it readable
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail}})


# -----------------------------
# Low-latency stream generators
# -----------------------------
async def kokoro_pcm_stream(text: str, voice: str, speed: float) -> AsyncGenerator[bytes, None]:
    """
    Streams S16LE PCM bytes immediately as Kokoro yields audio segments.
    Lowest latency output.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    chunks = chunk_text(text, WORD_THRESHOLD) if CHUNK_ENABLED else [text]

    for chunk in chunks:
        gen = pipeline(chunk, voice=voice, speed=speed, split_pattern=SPLIT_PATTERN)
        for (_gs, _ps, audio) in gen:
            a = as_numpy_float(audio)
            yield float_to_s16le_bytes(a)
            await asyncio.sleep(0)  # cooperative scheduling


async def mp3_stream_via_ffmpeg(
    pcm_gen: AsyncGenerator[bytes, None],
    sr: int,
    bitrate: str,
) -> AsyncGenerator[bytes, None]:
    """
    True streaming MP3:
    - Start ffmpeg encoder subprocess
    - Feed PCM chunks to stdin as they are generated
    - Yield MP3 frames from stdout as they are produced
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-i",
        "pipe:0",
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-f",
        "mp3",
        "-flush_packets",
        "1",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    async def feed_pcm():
        try:
            async for chunk in pcm_gen:
                proc.stdin.write(chunk)
                await proc.stdin.drain()
            try:
                proc.stdin.close()
            except Exception:
                pass
        except Exception:
            # If client disconnects, generator cancellation can happen
            try:
                proc.stdin.close()
            except Exception:
                pass

    feeder_task = asyncio.create_task(feed_pcm())

    try:
        while True:
            out = await proc.stdout.read(4096)
            if out:
                yield out
            else:
                break
            await asyncio.sleep(0)
    finally:
        # ensure feeding stops
        try:
            await feeder_task
        except Exception:
            pass

        # check ffmpeg result
        try:
            rc = await proc.wait()
        except Exception:
            rc = None

        if rc not in (0, None):
            try:
                err = await proc.stderr.read()
                logger.error(f"ffmpeg rc={rc}, err={err.decode('utf-8', errors='ignore')}")
            except Exception:
                logger.error(f"ffmpeg rc={rc}, stderr read failed")

        try:
            if proc.returncode is None:
                proc.kill()
        except Exception:
            pass


# -----------------------------
# OpenAI-compatible endpoint
# -----------------------------
@app.post("/v1/audio/speech")
async def audio_speech(req: OpenAISpeechRequest, request: Request):
    """
    OpenAI-compatible endpoint:
      POST /v1/audio/speech
    Returns raw audio bytes (streamed).
    """
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` is empty")

    voice = (req.voice or DEFAULT_VOICE).strip()
    speed = float(req.speed if req.speed is not None else DEFAULT_SPEED)
    fmt = (req.response_format or "mp3").lower()

    # OpenAI SDK sends Authorization: Bearer <api_key>. We ignore it.
    _auth = request.headers.get("authorization")

    start = time.perf_counter()
    logger.info(f"Request: fmt={fmt} voice={voice} speed={speed}")

    try:
        pcm_gen = kokoro_pcm_stream(text=text, voice=voice, speed=speed)

        if fmt in ("pcm", "pcm16", "s16", "raw"):
            # Absolute lowest latency, no encoder
            return StreamingResponse(pcm_gen, media_type=media_type_for(fmt))

        if fmt == "mp3":
            mp3_gen = mp3_stream_via_ffmpeg(pcm_gen=pcm_gen, sr=SR, bitrate=MP3_BITRATE)
            return StreamingResponse(mp3_gen, media_type=media_type_for(fmt))

        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")

    except HTTPException:
        raise
    except asyncio.CancelledError:
        logger.warning("Client disconnected (cancelled).")
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        total_ms = (time.perf_counter() - start) * 1000.0
        logger.info(f"Handler lifetime: {total_ms:.1f} ms")