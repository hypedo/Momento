"""HTTP 서버. 모델은 시작할 때 한 번 로드한다.

    cd momento_service && python -m uvicorn server:app --host 127.0.0.1 --port 8000

POST /generate  {"letter": "...", "event": "생일", "style": "pstl", "request_id": "optional"}
    -> image/png (헤더 X-Request-Id, X-Seed)
GET  /styles    -> 선택 가능한 화풍 / 기념일 목록
"""

from contextlib import asynccontextmanager
from io import BytesIO
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import config
from pipeline import MomentoPipeline

pipeline: MomentoPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = MomentoPipeline()
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    letter: str
    event: str
    style: str = "base"
    request_id: str | None = None


@app.get("/styles")
def styles():
    return {"styles": list(pipeline.styles), "events": config.EVENTS}


# async가 아닌 def라서 스레드풀에서 돈다 — 생성 중에도 다른 요청을 받고, 순서는 파이프라인 락이 정한다.
@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        result = pipeline.generate(req.letter, req.event, req.style, req.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    print(f"[{result.request_id}] {result.style} {result.timings}", flush=True)
    buf = BytesIO()
    result.image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        # HTTP 헤더는 latin-1만 허용해서 한글 request_id는 그대로 넣으면 응답 단계에서 터진다.
        headers={"X-Request-Id": quote(result.request_id), "X-Seed": str(result.seed)},
    )
