from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_heart",
    input="Hello world from Kokoro, true streaming mp3!",
    response_format="mp3",
) as resp:
    resp.stream_to_file("output.mp3")

print("Saved -> output.mp3")