FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    APP_ENTRYPOINT=main.py

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY . .

# Install dependencies with uv
RUN if [ -f requirements.txt ]; then \
        uv pip install --system -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
        uv pip install --system .; \
    fi

RUN addgroup --system app && adduser --system --ingroup app app && chown -R app:app ${APP_HOME}
USER app

CMD ["sh", "-c", "python ${APP_ENTRYPOINT}"]