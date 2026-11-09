"""1단계: 편지 -> 감정 분석 JSON (workspace: LLM_Emotion_Extraction/Qwen3-8B.py)"""

import json
import re

from llm import QwenLLM

MAX_ATTEMPTS = 3

INSTRUCTION = """
You are an expert in emotional understanding and literary analysis.

Your task is to deeply analyze a short personal letter.

Do not summarize the letter.
Instead, interpret the emotional meaning behind the text as deeply and precisely as possible.

Your primary goal is to extract rich and nuanced emotional information that can later be transformed into visual concepts for image generation.

Instructions:

1. Identify the primary and secondary emotions.
   CRITICAL: DO NOT use generic, basic words like "joy", "happiness", "sadness", or "love".
   Instead, use highly specific, nuanced emotional states (e.g., "unconditional supportive warmth", "teasing camaraderie", "nostalgic longing", "sweet modern affection", "energetic anticipation").

2. Identify both explicit emotions and implicit or hidden emotions.
   Do not limit the analysis to emotions that are directly stated.

3. For each emotion, explain why it exists based on the context and evidence in the letter.

4. Infer subtle emotions that the writer may not explicitly express.
   These may include feelings such as longing, nostalgia, relief, loneliness, affection, uncertainty, gratitude, regret, comfort, anticipation, or emotional attachment when supported by the text.

5. Analyze emotional nuances and mixed emotions.
   A letter may contain multiple emotions at the same time.
   Do not force the letter into a single simple emotional category.

6. Preserve subtle differences between similar letters.
   Do not produce identical emotional interpretations simply because two letters contain similar words or situations.
   Consider differences in context, wording, relationships, actions, and emotional tone.

7. Identify the emotional progression of the letter when applicable.
   Describe how the emotional state changes from the beginning to the end.

8. Infer the relationship or interpersonal context when it is relevant to understanding the emotions.
   Do not invent relationships or facts that are not reasonably supported by the text.

9. Describe the overall emotional meaning of the letter in a concise but nuanced way.

10. Describe the overall atmosphere and emotional tone.
    Make the atmosphere highly distinct for each letter. If the letter is a playful invitation (e.g., "come out tonight"), the atmosphere should be "bustling, dynamic, and energetic night". If it's a sweet gift message, it should be "cozy, pastel-toned, and relaxing". Avoid repeating the same "warm and bright" default.

11. Extract important emotional keywords.
    Prioritize words that represent emotional states, relationships, memories, experiences, or meaningful situations.

12. Estimate the overall emotional intensity on a scale from 1 to 10.

13. Do not invent unrelated details.
    Every interpretation should be reasonably supported by the letter.
    When an emotion is uncertain, represent that uncertainty through the confidence value.

14. Output only valid JSON.
    Do not include explanations, markdown, comments, or any text outside the JSON object.

Output format:

{
  "primary_emotion": "...",
  "secondary_emotions": ["...", "..."],

  "emotion_reason": [
    {
      "emotion": "...",
      "reason": "...",
      "evidence": "..."
    }
  ],

  "hidden_emotions": [
    {
      "emotion": "...",
      "confidence": 0.0,
      "reason": "..."
    }
  ],

  "emotional_nuances": ["...", "...", "..."],

  "emotional_progression": [
    {
      "stage": "...",
      "emotion": "...",
      "reason": "..."
    }
  ],

  "relationship_context": "...",

  "overall_emotional_meaning": "...",

  "atmosphere": {
    "overall": "...",
    "tone": "...",
    "pace": "...",
    "temperature": "...",
    "brightness": "..."
  },

  "keywords": ["...", "...", "..."],

  "emotion_intensity": 1
}
""".strip()


def extract_emotion(llm: QwenLLM, letter: str) -> str:
    """감정 분석 JSON 문자열을 돌려준다. 다음 단계에는 파싱 결과가 아니라 이 문자열이 그대로 들어간다."""
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": letter},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 한국어 인용이 많으면 1024 토큰에서 JSON이 잘려 파싱에 실패했다
        candidate = llm.chat(messages, max_new_tokens=2048, temperature=0.6, top_p=0.95)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError as e:
            # 가장 흔한 실패는 줄 끝 쉼표 누락이고 재생성해도 같은 자리에서 반복된다.
            # JSON 문자열에는 날 줄바꿈이 들어갈 수 없으므로, 값으로 끝난 줄 다음 줄이
            # 키/값으로 시작하면 쉼표를 넣어 다시 파싱한다.
            repaired = re.sub(r'(["\]}\d]|true|false|null)(\s*\n\s*)(["\[{])', r"\1,\2\3", candidate)
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                print(f"[emotion {attempt}/{MAX_ATTEMPTS}] invalid JSON: {e}", flush=True)

    raise RuntimeError(f"emotion analysis: no valid JSON after {MAX_ATTEMPTS} attempts")
