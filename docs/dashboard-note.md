# TradeX Frontend Dashboard Note

This frontend is intentionally lightweight and deployment-focused for Hugging Face Spaces.

## Scope

- React + Vite app lives in `frontend/`.
- Static build is served by Nginx in Docker.
- Container listens on `7860` to match Space expectations.

## Why this shape

- Keeps UI concerns separated from Python/OpenEnv runtime logic.
- Makes Space deploys deterministic (same build, same static output).
- Simplifies future upgrades (charts, API panels, telemetry pages) without touching backend internals.

## Next incremental additions

1. Add live benchmark cards fed by a JSON endpoint.
2. Add a run-history table using telemetry artifacts.
3. Add policy comparison charts with lightweight client-side plotting.
