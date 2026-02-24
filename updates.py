
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/kokoro_openai_compatible# docker logs 6c9106af6a69

==========
== CUDA ==
==========

CUDA Version 12.1.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Disabling PyTorch because PyTorch >= 2.4 is required but found 2.3.1+cu121
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
INFO:     Started server process [1]
INFO:     Waiting for application startup.
WARNING: Defaulting repo_id to hexgrad/Kokoro-82M. Pass repo_id='hexgrad/Kokoro-82M' to suppress this warning.
'[Errno 104] Connection reset by peer' thrown while requesting HEAD https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json
WARNING:huggingface_hub.utils._http:'[Errno 104] Connection reset by peer' thrown while requesting HEAD https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json
Retrying in 1s [Retry 1/5].
WARNING:huggingface_hub.utils._http:Retrying in 1s [Retry 1/5].
ERROR:    Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/kokoro/pipeline.py", line 113, in __init__
    self.model = KModel(repo_id=repo_id).to(device).eval()
  File "/usr/local/lib/python3.10/dist-packages/kokoro/model.py", line 46, in __init__
    config = hf_hub_download(repo_id=repo_id, filename='config.json')
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1032, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1165, in _hf_hub_download_to_cache_dir
    _get_metadata_or_catch_error(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1699, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1622, in get_hf_file_metadata
    response = _httpx_follow_relative_redirects(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 302, in _httpx_follow_relative_redirects
    response = http_backoff(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 506, in http_backoff
    return next(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 414, in _http_backoff_base
    response = client.request(method=method, url=url, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
  File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 901, in send
    raise RuntimeError("Cannot send a request, as the client has been closed.")
RuntimeError: Cannot send a request, as the client has been closed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 693, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 569, in __aenter__
    await self._router.startup()
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 670, in startup
    await handler()
  File "/app/app/main.py", line 127, in startup
    pipeline = KPipeline(lang_code=DEFAULT_LANG, device=device)
  File "/usr/local/lib/python3.10/dist-packages/kokoro/pipeline.py", line 116, in __init__
    raise RuntimeError(f"""Failed to initialize model on CUDA: {e}.
RuntimeError: Failed to initialize model on CUDA: Cannot send a request, as the client has been closed..
                                       Try setting device='cpu' or check CUDA installation.

ERROR:    Application startup failed. Exiting.
