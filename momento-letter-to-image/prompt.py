"""2단계: 편지 + 감정 분석 -> FLUX 프롬프트 (workspace: LLM_prompt_to_Diffusion/Qwen3-8B.py)"""

import json
import re
from dataclasses import dataclass

from llm import QwenLLM

MAX_ATTEMPTS = 5

CULTURAL_CONTEXT = (
    "Contemporary South Korean cultural context. "
    "Depict human subjects as Korean unless the original letter explicitly states otherwise."
)

# 재시도로도 글자 표면을 못 없앴을 때만 프롬프트 끝에 붙이는 안전망.
# 배경을 아웃포커스로 날려 간판 글자를 판독 불가능하게 만든다.
# 무조건 붙이면 멀쩡한 편지의 배경까지 뭉개지므로 폴백에서만 쓴다.
FRAMING_PROMPT = (
    "shallow depth of field, tight close-up on the subject, "
    "background heavily blurred into soft colored bokeh"
)

NEGATIVE_PROMPT = (
    "black and white, grayscale, monochrome, sepia, colorless image, "
    "photograph, photography, photorealistic, hyperrealistic, 3D render, "
    "extreme bleakness, oppressive gloom, horror atmosphere, "
    "readable text, letters, words, typography, captions, signs, "
    "logo, watermark, signature, "
    "blurry, low quality, distorted anatomy, malformed hands, "
    "extra fingers, missing fingers, extra limbs, missing limbs, "
    "fused bodies, duplicated person, duplicate face"
)

# 간판·화면 같은 글자 표면을 묘사하면 FLUX가 거기에 깨진 글자를 그려버린다.
# negative_prompt로는 막히지 않는다 — positive에 명시된 것은 CFG로 지워지지 않는다.
TEXT_PATTERN = re.compile(
    r"\b(?:signs?|signboards?|signage|banners?|posters?|menus?|billboards?|"
    r"labels?|captions?|logos?|typography|lettering|hangul)\b"
    r"|\bthat reads\b|\bin Korean\b|\btext messages?\b|\bwritten in\b",
    re.I,
)

# 단순 재생성은 모델이 장면 자체를 바꿔버려서(밤거리 -> 실내) 편지와 어긋났다.
# "장면은 유지하고 그 요소만 교체하라"고 지정해야 충실도가 보존된다.
FEEDBACK_TEMPLATE = (
    "Your visual_prompt contains {hits}. "
    "The image generator renders any such surface as unreadable broken glyphs. "
    "Rewrite the JSON: KEEP the exact same setting, time of day, action and mood, "
    "and change ONLY those elements — replace them with light, colour and texture "
    "that carry no writing (for example glowing lanterns, colored light, awnings, "
    "steam, foliage). Do not move the scene indoors or to a different location."
)

INSTRUCTION = """
You are an expert in visual storytelling, art direction, and text-to-image prompt engineering for SDXL.

Transform the CURRENT ORIGINAL LETTER and its EMOTION ANALYSIS into one concrete, situation-specific visual scene.

RULES:
1. FACTUAL ACCURACY: Depict ONLY the primary action, setting, and objects explicitly mentioned or directly implied in the CURRENT letter. Do not invent objects or mix elements from other contexts.
2. LETTER-SPECIFIC FOCUS: Make one concrete clue unique to the CURRENT letter the focal action. Do not default to the same cake, gift, or living-room scene for every letter.
3. EVENT CLARITY: Keep the requested event recognizable with one small supporting visual cue when needed; the letter-specific action must remain the focal point.
4. VISIBLE ACTIONS ONLY: Describe only physically renderable details that a camera can capture in a single frame. Do not use poetic, internal, or unseeable descriptions (e.g., "memories in eyes", "objects inside a closed box").
5. CONCRETE SDXL TERMS: Translate emotional tones into visible lighting, color palettes, and camera framing. Do not use abstract emotion words (e.g., "nostalgic", "loving", "sad").
6. CULTURAL CONTEXT & NO KOREAN TEXT: Depict human subjects as Korean in contemporary South Korea. NEVER include Korean characters (e.g., '생일축하', '사랑해') or text of any kind in the visual_prompt.
7. STYLE & FORMAT: Full-color and non-photographic. Do not include readable text, typography, logos, or watermarks.
8. DISTINCT PEOPLE: When more than one person appears, give each a different hairstyle, clothing color, and position in the frame (left/right, front/back) so no two look alike.
9. HIDE TEXT-BEARING SURFACES: Do not show phone or laptop screens, cards, letters, signs, book pages, or writing on cakes facing the viewer; show such objects closed, turned away, or held so their surface is hidden.

OUTPUT FORMAT:
Output ONLY valid JSON wrapped in ```json ... ``` codeblock with exactly these four keys:

{
  "physical_status": "together now | separated now | remembering the past | planning the future | not specified",
  "scene_summary": "one concise sentence describing the factual situation",
  "composition": "one concise sentence describing camera framing and lighting",
  "visual_prompt": "one dense 50-70 word visual prompt describing only physically observable content, subject, action, environment, and lighting"
}
""".strip()


@dataclass
class ScenePrompt:
    visual_prompt: str
    framing_applied: bool
    scene_summary: str


def strip_embedded_text(text: str) -> str:
    """따옴표로 감싼 문구나 한글을 지운다. LLM이 규칙 6을 가끔 어겨 깨진 한글이 그려지는 것을 막는다."""
    # 단어 사이의 '(woman's, it's)는 따옴표가 아니다. 이걸 따옴표로 보면 두 소유격 사이가 통째로 지워진다.
    text = re.sub(
        r"(?<!\w)[\"'‘’“”](?:[^\"'‘’“”]|(?<=\w)['’](?=\w))*[\"'‘’“”](?!\w)",
        "",
        text,
    )
    text = re.sub(r"[가-힣]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text.strip()


def _parse(candidate: str) -> dict:
    if "```json" in candidate:
        candidate = candidate.split("```json")[1].split("```")[0]
    elif "```" in candidate:
        candidate = candidate.split("```")[1]
    return json.loads(candidate.strip())


def build_scene(llm: QwenLLM, letter: str, emotion: str) -> ScenePrompt:
    user_content = (
        f"CURRENT ORIGINAL LETTER — factual ground truth:\n{letter}\n\n"
        f"EMOTION ANALYSIS — emotional guidance only:\n{emotion}\n\n"
        f"CULTURAL CONTEXT — apply only when not contradicted by the letter:\n{CULTURAL_CONTEXT}"
    )
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    fallback = None  # JSON은 유효하지만 글자 표면이 남아 있는 결과

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 384: visual_prompt 50-70단어 + 나머지 키. 256이면 JSON이 잘릴 수 있다
        candidate = llm.chat(messages, max_new_tokens=384, temperature=0.7, top_p=0.9)
        try:
            parsed = _parse(candidate)
        except json.JSONDecodeError as e:
            print(f"[prompt {attempt}/{MAX_ATTEMPTS}] invalid JSON: {e}", flush=True)
            continue

        visual_prompt = strip_embedded_text(parsed.get("visual_prompt", ""))
        if not visual_prompt:
            print(f"[prompt {attempt}/{MAX_ATTEMPTS}] empty visual_prompt", flush=True)
            continue

        hits = [m.group(0) for m in TEXT_PATTERN.finditer(visual_prompt)]
        if not hits:
            return ScenePrompt(visual_prompt, False, parsed.get("scene_summary", ""))

        print(f"[prompt {attempt}/{MAX_ATTEMPTS}] text surfaces {hits}; asking for a rewrite", flush=True)
        if fallback is None:
            fallback = ScenePrompt(visual_prompt, True, parsed.get("scene_summary", ""))
        messages.append({"role": "assistant", "content": candidate})
        messages.append({
            "role": "user",
            "content": FEEDBACK_TEMPLATE.format(hits=", ".join(repr(h) for h in hits)),
        })

    if fallback is not None:
        print("[prompt] text surfaces remain; applying framing fallback", flush=True)
        return fallback
    raise RuntimeError(f"prompt generation: no valid JSON after {MAX_ATTEMPTS} attempts")


def compose_flux_prompt(event: str, trigger: str, scene: ScenePrompt) -> str:
    # 이벤트 -> 화풍 트리거 -> 장면 묘사 순서. 구도 수식어는 트리거와 섞이지 않게 맨 뒤.
    prompt = ", ".join(part for part in (event, trigger, scene.visual_prompt) if part)
    if scene.framing_applied:
        prompt = f"{prompt}, {FRAMING_PROMPT}"
    return prompt
