from __future__ import annotations

from openenv.core.env_client.http_client import EnvClient

try:
    from .models import PIIAction, PIIObservation, PIIState
except ImportError:
    from models import PIIAction, PIIObservation, PIIState


class PIIRedactionEnvClient(EnvClient[PIIAction, PIIObservation, PIIState]):
    """
    Client for the PII Redaction environment.

    Usage (sync):
        with PIIRedactionEnvClient(base_url="https://Clueless13-pii-redaction-env.hf.space").sync() as client:
            obs = client.reset()
            result = client.step(PIIAction(spans=[], submit=True))

    Usage (async):
        async with PIIRedactionEnvClient(base_url="https://Clueless13-pii-redaction-env.hf.space") as client:
            obs = await client.reset()
            result = await client.step(PIIAction(spans=[], submit=True))
    """

    action_type = PIIAction
    observation_type = PIIObservation
    state_type = PIIState
