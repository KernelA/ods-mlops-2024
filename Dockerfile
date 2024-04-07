ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-buster as builder

WORKDIR /home/app

ARG PDM_CACHE=/opt/pdm/cache

ENV PDM_CACHE_DIR=${PDM_CACHE}

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip\
    pip install -U pip wheel && \
    pip install pdm

COPY ./pyproject.toml ./pdm.lock ./

# Add adiitional instructions
RUN --mount=type=cache,id=pip-cache,target=${PDM_CACHE} \
    pdm install --prod --no-self --frozen-lockfile

COPY ./README.md ./

COPY ./ods_mlops ./ods_mlops

RUN --mount=type=cache,id=pip-cache,target=${PDM_CACHE} \
    pdm install --prod --frozen-lockfile --no-editable

FROM python:${PYTHON_VERSION}-slim

ARG PROJECT_VENV_DIR=/home/app/.venv

COPY --from=builder ${PROJECT_VENV_DIR} ${PROJECT_VENV_DIR}

ENV PATH="${PROJECT_VENV_DIR}/bin:$PATH"

ENV TZ="Europe/Moscow"

WORKDIR /home/app

ENV LOG_CONFIG=log_settings.yaml

COPY ./main.py ./log_settings.yaml ./

CMD python ./main.py
