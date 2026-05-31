from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .prompts import PAGE_ANALYSIS_PROMPT


@dataclass
class LocalQwenVL:
    """Minimal local Qwen2.5-VL wrapper for single-page paper parsing."""

    model_path: str
    device: str = "cuda"
    max_new_tokens: int = 768
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        if not self.device.startswith("cuda"):
            self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def analyze_page(self, image_path: Path, prompt: str = PAGE_ANALYSIS_PROMPT) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils is required. Run `pip install -r requirements.txt`.") from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.strip()


def load_local_qwen_vl() -> LocalQwenVL:
    return LocalQwenVL(
        model_path=os.getenv("VISION_MODEL", "./models/Qwen2.5-VL-3B-Instruct"),
        device=os.getenv("VISION_DEVICE", "cuda"),
        max_new_tokens=int(os.getenv("VISION_MAX_NEW_TOKENS", "768")),
        min_pixels=int(os.getenv("VISION_MIN_PIXELS", str(256 * 28 * 28))),
        max_pixels=int(os.getenv("VISION_MAX_PIXELS", str(1280 * 28 * 28))),
    )
