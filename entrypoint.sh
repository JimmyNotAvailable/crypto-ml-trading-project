#!/bin/bash
# Entrypoint script for crypto project
set -euo pipefail

SERVICE_MODE=${SERVICE:-web}

echo "🚀 Starting service: ${SERVICE_MODE}"

if [ "$SERVICE_MODE" = "web" ]; then
	exec python web/app.py
elif [ "$SERVICE_MODE" = "bot" ]; then
	if [ -z "${BOT_TOKEN:-}" ] && [ -f "/app/token.txt" ]; then
		echo "ℹ️ Using token from token.txt"
		export BOT_TOKEN=$(head -n1 /app/token.txt)
	fi
	exec python -m app.bot
elif [ "$SERVICE_MODE" = "demo" ]; then
	# Run both web and bot (bot optional)
	set +e
	(python -m app.bot &) 
	set -e
	if [ "${DASHBOARD:-0}" = "1" ]; then
		exec python examples/ml/web_dashboard.py
	else
		exec python web/app.py
	fi
else
	echo "❌ Unknown SERVICE mode: $SERVICE_MODE" >&2
	exit 1
fi
