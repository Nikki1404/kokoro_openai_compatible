
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 8, in <module>
    with client.audio.speech.with_streaming_response.create(
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model="kokoro",
        ^^^^^^^^^^^^^^^
        voice="af_heart",
        ^^^^^^^^^^^^^^^^^
        input="Hello from Kokoro using KPipeline",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as resp:
    ^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_response.py", line 626, in __enter__
    self.__response = self._request_func()
                      ~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\resources\audio\speech.py", line 104, in create
    return self._post(
           ~~~~~~~~~~^
        "/audio/speech",
        ^^^^^^^^^^^^^^^^
    ...<15 lines>...
        cast_to=_legacy_response.HttpxBinaryResponseContent,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1037, in request
    raise APIConnectionError(request=request) from err
openai.APIConnectionError: Connection error.
