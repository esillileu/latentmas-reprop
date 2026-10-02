import sys

from .cli import parse_run_matrix
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for src.run module."""
    runs = parse_run_matrix(argv)
    if runs[0].dry_run:
        varying = [
            key
            for key in vars(runs[0])
            if len({repr(getattr(run, key)) for run in runs}) > 1
        ]
        print(f"Run plan: {len(runs)} run(s)")
        for index, run in enumerate(runs, start=1):
            details = ", ".join(f"{key}={getattr(run, key)}" for key in varying)
            if run.receiver_reasoning:
                details = (
                    f"model_name={run.model_name}, "
                    f"upstream_steps={[0, *run.upstream_steps]}, "
                    f"receiver_modes=answer_only/free, max_samples={run.max_samples}"
                )
            if run.receiver_compute_preflight:
                details = (
                    f"model={run.model_name}, U={run.upstream_steps}, "
                    f"M=matched/mismatched/no_handoff, R={run.receiver_budgets}, "
                    f"free_cap={run.max_new_tokens}, samples={run.max_samples}"
                )
            print(f"[{index}/{len(runs)}] {details or 'single configuration'}")
        return
    for args in runs:
        run_benchmark(args)


if __name__ == "__main__":
    main(sys.argv[1:])
