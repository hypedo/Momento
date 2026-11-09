"""3단계: 프롬프트 -> 이미지 (workspace: Diffusion/FLUX.py)"""

import gc
from pathlib import Path

import torch
from PIL import Image


class FluxGenerator:
    def __init__(self, model_id: str, device: str, lora_dirs: dict[str, Path]):
        # FLUX.2-klein은 Flux2Pipeline이 아니라 Flux2KleinPipeline(Qwen3 텍스트 인코더용)을 써야 한다.
        # Flux2Pipeline으로도 예외 없이 로드되지만 generate() 시점에야 chat template 오류로 죽는다.
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device)
        self.pipe.vae.enable_slicing()

        # 화풍 LoRA는 시작할 때 전부 올려두고 요청마다 set_adapters로 고른다.
        for name, path in lora_dirs.items():
            self.pipe.load_lora_weights(str(path), adapter_name=name)
        self._has_loras = bool(lora_dirs)
        self._negative_cache: dict[str, torch.Tensor] = {}

    def _negative_embeds(self, negative_prompt: str) -> torch.Tensor:
        # 이 파이프라인의 __call__은 negative_prompt 문자열을 받지 않고 negative_prompt_embeds만 받으므로
        # 직접 인코딩해 넘겨야 실제로 적용된다. __call__ 밖에서 encode_prompt를 부르면 grad tracking이
        # 켜진 채 4B 텍스트 인코더 활성값을 저장해 VRAM이 터지므로 no_grad가 필요하다.
        if negative_prompt not in self._negative_cache:
            with torch.no_grad():
                embeds = self.pipe.encode_prompt(prompt=negative_prompt, device=self.pipe._execution_device)[0]
            self._negative_cache[negative_prompt] = embeds.cpu()
        return self._negative_cache[negative_prompt].to(self.pipe._execution_device)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        style: str,
        lora_scale: float,
        guidance: float,
        seed: int,
        width: int,
        height: int,
        steps: int,
    ) -> Image.Image:
        if self._has_loras:
            if style == "base":
                self.pipe.disable_lora()
            else:
                self.pipe.enable_lora()
                self.pipe.set_adapters([style], adapter_weights=[lora_scale])
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }
        # klein-base는 guidance-distilled가 아니라서 guidance > 1이면 실제 CFG를 수행한다.
        if negative_prompt and guidance > 1.0:
            kwargs["negative_prompt_embeds"] = self._negative_embeds(negative_prompt)

        image = self.pipe(**kwargs).images[0]

        # 계속 켜놓고 연달아 생성할 때 CUDA 캐시가 쌓여 VRAM이 터지는 것을 막는다.
        gc.collect()
        torch.cuda.empty_cache()
        return image
