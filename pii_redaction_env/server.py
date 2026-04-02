from __future__ import annotations

import uvicorn
from openenv.core.env_server.http_server import create_app

try:
    from .env import PIIRedactionEnv
    from .models import PIIAction, PIIObservation
except ImportError:
    from env import PIIRedactionEnv
    from models import PIIAction, PIIObservation


app = create_app(
    PIIRedactionEnv,
    PIIAction,
    PIIObservation,
    env_name="pii_redaction_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
