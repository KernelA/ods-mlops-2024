import enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(enum.Enum):
    PENDING = "pending"
    STARTED = "started"
    RETRY = "retry"
    FAILUR = "failure"
    SUCCESS = "success"


class SentimentType(enum.Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class InputText(BaseModel):
    text: str = Field(min_length=1, max_length=500)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "ты супер!",
                },
            ]
        }
    }


class SentimentResult(BaseModel):
    label: SentimentType
    score: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"label": "positive", "score": 0.8923923},
            ]
        }
    }


class Task(BaseModel):
    task_id: str
    task_status: TaskStatus
    task_result: Optional[SentimentResult] = None
