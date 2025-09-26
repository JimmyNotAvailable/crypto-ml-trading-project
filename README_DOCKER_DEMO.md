# Crypto ML Project — Docker Demo

This guide shows how to run the current web dashboard and Discord bot via Docker Compose.

## Prerequisites
- Docker Desktop 4.x+
- A Discord Bot token (optional for web-only). Create `.env` from `.env.example` or set env var at runtime.

## Services
- `web`: Flask app, exposed on WEB_PORT (default 8000)
- `bot`: Discord bot (requires `BOT_TOKEN`)
- `demo`: Runs both web and bot together
- `mongo`: MongoDB for persistence

## Quick start

1) Web only
```
docker compose up -d mongo web
```
Visit http://localhost:8000

2) Bot only (requires token)
```
# Windows PowerShell
$env:BOT_TOKEN="<your-token>"; docker compose up -d mongo bot
```

3) Full demo (web + bot)
```
# Windows PowerShell
$env:BOT_TOKEN="<your-token>"; docker compose up -d mongo demo
```

Logs:
```
docker compose logs -f web
# or
docker compose logs -f bot
```

## Bot commands
- `!ping` — health check
- `!help` — list commands
- `!predict` — price prediction (uses production models if available, else a stub)
- `!predict_json {json}` — send custom features as JSON

## Notes
- If `data/models_production/crypto_models_production.pkl` exists, the bot will use it via `data/models_production/quick_loader.py`. Otherwise, it uses a safe demo stub.
- The entrypoint supports SERVICE modes: `web` (default), `bot`, or `demo`.
- `WEB_PORT` can be overridden in `.env` or via environment at runtime.
