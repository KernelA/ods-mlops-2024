from celery.result import AsyncResult
from fastapi import APIRouter

from ..models import InputText, SentimentResult, SentimentType, Task, TaskStatus
from ..worker import analyze

router = APIRouter()


@router.post("/analyze/send", response_model=Task)
async def analyze_text(request: InputText):
    task = analyze.delay(text=request.text)
    return Task(task_id=task.id, task_status=task.status.lower(), task_result=None)


@router.get("/analyze/status/{task_id}", response_model=Task)
async def analyze_status(task_id: str):
    task: AsyncResult = analyze.app.AsyncResult(task_id)

    result = None
    task_status = TaskStatus[task.status]

    if task_status == TaskStatus.SUCCESS:
        result = SentimentResult(
            label=SentimentType[task.result["label"].upper()],
            score=task.result["score"],
        )

    return Task(task_id=task.id, task_status=task_status, task_result=result)
