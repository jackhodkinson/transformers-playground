# Transformers Playground

This is a little playground for myself learning how to use transformers on Apple Silicon.

I found it quite frustrating to get DeepSeek Coder (`deepseek-ai/DeepSeek-Coder-V2-Lite-Base`) working on Apple Silicon. It required version pinning on advice from [this issue](https://github.com/OpenBMB/MiniCPM-o/issues/722#issuecomment-2592260410), and a monkey patch to remove the `flash_attn` dependency on advice from [this discussion](https://huggingface.co/qnguyen3/nanoLLaVA-1.5/discussions/4). Now it seems to work ok. Run `uv run scripts/deepseek.py` to see it in action.
