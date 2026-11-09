"""편지와 기념일을 장면으로 변환해 FLUX로 이미지를 생성한다."""

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field

from PIL import Image

import config
from emotion import extract_emotion
from flux import FluxGenerator
from llm import QwenLLM
from prompt import NEGATIVE_PROMPT, build_scene, compose_flux_prompt


@dataclass
class Result:
    image: Image.Image
    request_id: str
    event: str
    style: str
    seed: int
    flux_prompt: str
    scene_summary: str
    framing_applied: bool
    emotion: str
    timings: dict = field(default_factory=dict)


def resolve_event(event: str) -> str:
    event = event.strip()
    return config.EVENTS.get(event, config.EVENTS.get(event.lower(), event))


def seed_from_id(request_id: str) -> int:
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)


class MomentoPipeline:
    def __init__(self):
        self.llm = QwenLLM(config.LLM_MODEL, config.DEVICE)
        print("loading FLUX.2-klein + style LoRAs ...", flush=True)
        self.styles = config.available_styles()
        lora_dirs = {name: config.LORA_DIR / name for name in self.styles if name != "base"}
        self.flux = FluxGenerator(config.FLUX_MODEL, config.DEVICE, lora_dirs)
        # GPU가 하나라 요청은 한 번에 하나씩 처리한다.
        self._lock = threading.Lock()
        print("pipeline ready", flush=True)

    def generate(self, letter: str, event: str, style: str, request_id: str | None = None) -> Result:
        letter = letter.strip()
        if not letter:
            raise ValueError("letter is empty")
        if style not in self.styles:
            raise ValueError(f"style '{style}' is unavailable. choose one of {list(self.styles)}")

        event = resolve_event(event)
        request_id = request_id or uuid.uuid4().hex
        preset = self.styles[style]
        seed = seed_from_id(request_id)
        timings = {}

        with self._lock:
            t = time.time()
            self.llm.load()
            timings["llm_load_sec"] = round(time.time() - t, 1)
            try:
                t = time.time()
                emotion = extract_emotion(self.llm, letter)
                timings["emotion_sec"] = round(time.time() - t, 1)

                t = time.time()
                scene = build_scene(self.llm, letter, emotion)
                timings["prompt_sec"] = round(time.time() - t, 1)
            finally:
                self.llm.unload()

            flux_prompt = compose_flux_prompt(event, preset["trigger"], scene)
            t = time.time()
            image = self.flux.generate(
                prompt=flux_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                style=style,
                lora_scale=config.LORA_SCALE,
                guidance=preset["guidance"],
                seed=seed,
                width=config.IMAGE_WIDTH,
                height=config.IMAGE_HEIGHT,
                steps=config.STEPS,
            )
            timings["image_sec"] = round(time.time() - t, 1)

        return Result(
            image=image,
            request_id=request_id,
            event=event,
            style=style,
            seed=seed,
            flux_prompt=flux_prompt,
            scene_summary=scene.scene_summary,
            framing_applied=scene.framing_applied,
            emotion=emotion,
            timings=timings,
        )
