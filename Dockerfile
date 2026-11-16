FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash apex

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apex_quant/ ./apex_quant/
COPY main.py pyproject.toml ./

RUN mkdir -p /app/data && chown -R apex:apex /app
USER apex

ENV APEX_LOCAL_STATE_FILE=/app/data/apex_state_v19.json

ENTRYPOINT ["python", "main.py"]
CMD ["run", "--loop", "--interval", "300"]
