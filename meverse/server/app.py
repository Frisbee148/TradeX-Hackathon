"""FastAPI application for the market surveillance environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import SurveillanceAction, SurveillanceObservation
    from .meverse_environment import MarketSurveillanceEnvironment
except (ModuleNotFoundError, ImportError, ValueError):
    from models import SurveillanceAction, SurveillanceObservation
    from server.meverse_environment import MarketSurveillanceEnvironment


app = create_app(
    MarketSurveillanceEnvironment,
    SurveillanceAction,
    SurveillanceObservation,
    env_name="amm-market-surveillance",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
