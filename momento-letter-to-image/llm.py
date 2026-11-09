import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLLM:
    """감정 분석과 프롬프트 생성이 같은 Qwen3-8B를 쓰므로 요청마다 한 번 올려 두 단계가 공유한다.

    컨테이너 RAM 한도가 ~34GB라 Qwen3-8B(16GB)를 CPU에 상주시키면 FLUX(16GB)와 합쳐 OOM으로 죽는다.
    그래서 상주시키지 않고 요청마다 GPU로 바로 로드했다가 FLUX 단계 전에 완전히 내린다.
    """

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self.model.eval()

    def unload(self) -> None:
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int, temperature: float, top_p: float) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        output_ids = generated[0][len(inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
