from transformers import pipeline

from ..settings import get_settings

if __name__ == "__main__":
    pipeline(model=get_settings().cls_model_name)
