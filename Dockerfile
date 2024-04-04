ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-buster as builder

WORKDIR /home/app

RUN python -m venv /opt/venv

ENV PATH=/opt/venv/bin:${PATH}

RUN pip install -U pip setuptools wheel && \
    pip install pdm

# Add adiitional instructions
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=pdm.lock,target=pdm.lock \
    pdm install --prod --no-self --frozen-lockfile

RUN --mount=type=bind,source=./ods_mlops,target./ods_mlops \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=pdm.lock,target=pdm.lock \
    pdm install --prod --frozen-lockfile --no-editable

FROM python:${PYTHON_VERSION}-slim

ENV TZ="Europe/Moscow"

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:${PATH}

WORKDIR /home/app

COPY ./main.py ./
