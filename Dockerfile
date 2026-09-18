FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY src ./src
COPY data/sample ./data/sample
# Editable install keeps PROJECT_ROOT at /app so config paths resolve to /app/data, /app/models.
RUN pip install --no-deps -e .

RUN useradd --create-home app && mkdir -p data/raw data/processed models && chown -R app /app
USER app

EXPOSE 8501
CMD ["streamlit", "run", "src/tram_mlops/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
