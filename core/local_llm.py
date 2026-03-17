from __future__ import annotations

from typing import Optional


class LocalLLM:
    """
    Very small wrapper around a local Transformers chat model.
    Loads lazily on first use.
    """

    _tokenizer = None
    _model = None
    _model_name = None
    _device = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        import os

        self.model_name = model_name or os.getenv("AI_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    def _ensure_loaded(self) -> None:
        if self.__class__._model is not None and self.__class__._model_name == self.model_name:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        cuda_available = torch.cuda.is_available()
        device = "cuda:0" if cuda_available else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            device_map=device,
        )
        model.eval()

        self.__class__._tokenizer = tokenizer
        self.__class__._model = model
        self.__class__._model_name = self.model_name
        self.__class__._device = device

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.0) -> str:
        self._ensure_loaded()

        tokenizer = self.__class__._tokenizer
        model = self.__class__._model
        device = self.__class__._device

        conversation = [
            {"role": "system", "content": "You are a strict JSON generator. Output JSON only."},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        import torch

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0,
                temperature=float(temperature) if temperature > 0 else None,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

