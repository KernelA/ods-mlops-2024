ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-buster as builder

WORKDIR /home/app

ARG PDM_CACHE=/opt/pdm/cache

ENV PDM_CACHE_DIR=${PDM_CACHE}

RUN --mount=type=cache,target=/root/.cache/pip\
    pip install -U pip wheel && \
    pip install pdm~=2.15.0

COPY ./pyproject.toml ./pdm.lock ./

RUN --mount=type=cache,id=pdm-cache,target=${PDM_CACHE} \
    pdm install --prod --no-self --with web --frozen-lockfile

COPY ./README.md ./

COPY ./ods_mlops ./ods_mlops

RUN --mount=type=cache,id=pdm-cache,target=${PDM_CACHE} \
    pdm install --prod --with web --frozen-lockfile --no-editable

FROM python:${PYTHON_VERSION}-slim

ARG PROJECT_VENV_DIR=/home/app/.venv

COPY --from=builder ${PROJECT_VENV_DIR} ${PROJECT_VENV_DIR}

ENV PATH="${PROJECT_VENV_DIR}/bin:$PATH"

ARG HF_HOME=/opt/hf-cache

ENV HF_HOME=${HF_HOME}

RUN --mount=type=cache,target=${HF_HOME} \
    python -m ods_mlops.web.download_model

ENV TZ="Europe/Moscow"

WORKDIR /home/app

ENV LOG_CONFIG=log_settings.yaml

COPY ./main.py ./log_settings.yaml ./

CMD python ./main.py
