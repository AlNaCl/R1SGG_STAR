"""Pick Qwen2-VL vs Qwen2.5-VL class from checkpoint config (model_type)."""
from __future__ import annotations

import torch
from transformers import AutoConfig, Qwen2VLForConditionalGeneration


def load_qwen_vl_for_inference(
    model_path: str,
    *,
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype | None = None,
    device_map: str | dict = "cuda",
    trust_remote_code: bool = True,
):
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    kwargs = dict(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    mt = getattr(config, "model_type", None)
    if mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(**kwargs)
    return Qwen2VLForConditionalGeneration.from_pretrained(**kwargs)
