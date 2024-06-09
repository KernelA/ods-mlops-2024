from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_APP_")
    app_name: str = "Sentiment aanlyze"
    broker_url: str = "amqp://user:pass@localhost:5672/host"
    backend_url: str = "redis://localhost"
    cls_model_name: str = "seara/rubert-tiny2-russian-sentiment"


@lru_cache
def get_settings():
    return Settings()
