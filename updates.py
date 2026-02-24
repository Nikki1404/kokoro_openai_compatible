  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 8, in <module>
    response = client.audio.speech.create(
        model="kokoro",
    ...<2 lines>...
        response_format="mp3",
    )
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
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1070, in request
    raise self._make_status_error_from_response(err.response) from None
openai.InternalServerError: Internal Server Error
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 8, in <module>
    response = client.audio.speech.create(
        model="kokoro",
    ...<2 lines>...
        response_format="mp3",
    )
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
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1070, in request
    raise self._make_status_error_from_response(err.response) from None
openai.InternalServerError: Internal Server Error
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Saved -> output_fixed1.mp3
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 16, in <module>
    f.write(response.content)
            ^^^^^^^^
NameError: name 'response' is not defined
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Saved -> output_fixed1.wav
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> cd ..
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl> cd app
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_transports\default.py", line 101, in map_httpcore_exceptions
    yield
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_transports\default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_sync\connection_pool.py", line 256, in handle_request
    raise exc from None
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_sync\connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_sync\connection.py", line 101, in handle_request
    raise exc
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_sync\connection.py", line 78, in handle_request
    stream = self._connect(request)
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_sync\connection.py", line 124, in _connect
    stream = self._network_backend.connect_tcp(**kwargs)
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_backends\sync.py", line 207, in connect_tcp
    with map_exceptions(exc_map):
         ~~~~~~~~~~~~~~^^^^^^^^^
  File "C:\Program Files\Python313\Lib\contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpcore\_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1005, in request
    response = self._client.send(
        request,
        stream=stream or self._should_stream_response_body(request=request),
        **kwargs,
    )
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_transports\default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Program Files\Python313\Lib\contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\httpx\_transports\default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it

The above exception was the direct cause of the following exception:

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
getting this 
