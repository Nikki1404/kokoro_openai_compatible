from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="not-needed",   # ignored by your server
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",                 # or "tts-1" (ignored by server)
    voice="af_sky",                 # or "af_heart"
    input="Hello world from Kokoro!",
    response_format="mp3",          # "mp3" or "pcm"
) as resp:
    resp.stream_to_file("output.mp3")

print("Saved -> output.mp3")
