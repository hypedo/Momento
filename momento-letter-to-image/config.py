"""서비스 설정. 값은 flux_lora-workspace에서 검증된 설정을 그대로 옮겨온 것이다."""

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
LORA_DIR = SERVICE_DIR / "loras"
LORA_FILENAME = "pytorch_lora_weights.safetensors"

DEVICE = "cuda:0"

LLM_MODEL = "Qwen/Qwen3-8B"
FLUX_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
STEPS = 28
LORA_SCALE = 1.0

# 화풍 LoRA. guidance는 LoRA마다 다르다 — LoRA에 CFG 3.5를 걸면 negative의
# grayscale/colorless 반대 방향으로 과하게 밀려 학습 데이터보다 채도가 진하게 나온다.
STYLES = {
    "base":    {"trigger": "", "guidance": 3.5},
    "pstl":    {"trigger": "mmt_pstl",  "guidance": 1.5},
    "water":   {"trigger": "mmt_wtrc",  "guidance": 2.5},
    "cozy":    {"trigger": "mmt_cozy",  "guidance": 1.5},
    "vintage": {"trigger": "mmt_vntg",  "guidance": 2.5},
    "anime":   {"trigger": "mmt_anime", "guidance": 2.5},
    "josen":   {"trigger": "mmt_josn",  "guidance": 3.5},
}

def available_styles() -> dict:
    return {
        name: preset
        for name, preset in STYLES.items()
        if name == "base" or (LORA_DIR / name / LORA_FILENAME).is_file()
    }


# 기념일 입력 -> 프롬프트 맨 앞에 붙는 영어 이벤트 문구.
# 여기 없는 값은 영어 문구로 보고 그대로 쓴다.
EVENTS = {
    "birthday": "a birthday celebration",
    "christmas": "a Christmas celebration",
    "anniversary": "a couple's anniversary",
    "생일": "a birthday celebration",
    "크리스마스": "a Christmas celebration",
    "기념일": "a couple's anniversary",
    "연인 기념일": "a couple's anniversary",
}
