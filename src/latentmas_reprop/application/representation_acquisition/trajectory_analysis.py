"""Offline pairing of receiver trajectories with previously analyzed sender probes."""

from .sampling import generate_secret_digit_samples


def summarize_trajectory(records, probe_cells, model, expected_samples=100):
    """Join paired prediction changes to saved full20 post-realignment probes."""
    drops = {r["sample_id"]: r for r in records if r["condition"] == "drop"}
    if len(drops) != expected_samples:
        raise ValueError("Incomplete or duplicate drop sample set")
    if len(records) != expected_samples * 21 or any(r["error"] for r in records):
        raise ValueError("Incomplete receiver trajectory")
    canonical = {s.sample_id: s for s in generate_secret_digit_samples(100, 42)}
    for row in records:
        sample = canonical.get(row["sample_id"])
        if sample is None or (row["sample_key"], row["target_digit"]) != (
            sample.sample_key,
            sample.digit,
        ):
            raise ValueError("Receiver sample is not from the canonical balanced 100")
        if (
            row["model"] != model
            or row["seed"] != 42
            or row["predicted_digit"] not in range(10)
        ):
            raise ValueError("Incorrect model/seed or invalid receiver prediction")
    probes = [
        r
        for r in probe_cells
        if r["model"] == model
        and r["latent_steps"] == "20"
        and r["representation"] == "latent_post_realign"
    ]
    if len(probes) != 20 or {int(r["step"]) for r in probes} != set(range(1, 21)):
        raise ValueError("Expected unique saved post-realignment probe cells 1..20")
    result = []
    for probe in sorted(probes, key=lambda r: int(r["step"])):
        step = int(probe["step"])
        own = [
            r for r in records if r["condition"] == "own" and r["latent_steps"] == step
        ]
        if len(own) != expected_samples or {r["sample_id"] for r in own} != set(drops):
            raise ValueError(f"Unpaired or duplicate receiver samples at step {step}")
        for r in own:
            baseline = drops[r["sample_id"]]
            if (r["sample_key"], r["target_digit"]) != (
                baseline["sample_key"],
                baseline["target_digit"],
            ):
                raise ValueError("Mismatched drop identity")
            if (
                r["context_mode"] != "latent_only"
                or r["source_digit"] != r["target_digit"]
            ):
                raise ValueError("Expected latent-only own handoff")
            if (
                r["cache_sequence_length"] != step
                or r["receiver_position_start"] != r["original_full_seq_len"]
                or r["retained_tail_start_position"] + step
                != r["original_full_seq_len"]
                or r["retained_original_position_end"] != r["original_full_seq_len"]
            ):
                raise ValueError("Incorrect handoff length or receiver position")
        changed = sum(
            r["predicted_digit"] != drops[r["sample_id"]]["predicted_digit"]
            for r in own
        )
        result.append(
            {
                "model": model,
                "latent_step": step,
                "sample_count": expected_samples,
                "probe_run_id": probe["run_id"],
                "probe_accuracy": float(probe["probe_accuracy"]),
                "probe_null_mean": float(probe["null_mean_accuracy"]),
                "probe_fwer_p": float(probe["fwer_p_value"]),
                "probe_effect_pp": 100
                * (float(probe["probe_accuracy"]) - float(probe["null_mean_accuracy"])),
                "probe_significant": probe["significance"].lower() == "true",
                "receiver_changed_fraction": changed / expected_samples,
            }
        )
    return result
