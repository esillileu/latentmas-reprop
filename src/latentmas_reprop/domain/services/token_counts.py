"""Separate decoded text tokens from latent rollout steps."""


def attach_token_counts(result: dict) -> dict:
    """Store text-token and latent-step totals on a sample result."""
    agents = result.get("agents") or []
    result["generated_tokens"] = sum(
        int(agent.get("generated_tokens") or 0) for agent in agents
    )
    result["latent_steps_executed"] = sum(
        int(agent["latent_steps"])
        for agent in agents
        if isinstance(agent.get("latent_steps"), int)
    )
    return result
