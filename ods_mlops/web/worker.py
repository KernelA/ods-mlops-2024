from celery import Celery, Task
from celery.signals import worker_process_init
from transformers import pipeline

from .settings import get_settings

setting = get_settings()


class SentimentCls(Task):
    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self._pipeline = None
        worker_process_init.connect(self._init_model)

    def _init_model(self, **kwargs):
        if self._pipeline is None:
            self._pipeline = pipeline(model=self._model_name)

    def run(self, text: str):
        return self._pipeline(text)[0]


celery_app = Celery(__name__)
celery_app.conf.broker_url = setting.broker_url
celery_app.conf.result_backend = setting.backend_url


analyze = celery_app.register_task(SentimentCls(setting.cls_model_name))
