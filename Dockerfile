FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
 && apt-get install --no-install-recommends -y bash ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -u 10012 -m -d /home/appuser appuser \
 && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . /app
RUN chmod +x /app/entrypoint.sh

USER appuser

RUN python -m pip install --no-cache-dir --user -r requirements.txt

ENTRYPOINT ["/app/entrypoint.sh"]
