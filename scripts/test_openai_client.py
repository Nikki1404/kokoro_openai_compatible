from openai import OpenAI

# 🔥 CHANGE THIS depending on where server runs
# If local:
BASE_URL = "http://localhost:8080/v1"

# If remote Linux server:
# BASE_URL = "http://<SERVER_IP>:8080/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="not-needed",
)

# Use streaming (required for your implementation)
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_heart",
    input="Hello from Kokoro using KPipeline",
    response_format="mp3",   # "mp3" or "pcm"
) as resp:

    # Save streamed file
    resp.stream_to_file("output.mp3")

print("Saved -> output.mp3")
