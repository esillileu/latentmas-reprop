"""Receiver compute preflight options and constraints."""


def add_compute_preflight_args(parser, defaults):
    parser.add_argument(
        "--receiver_compute_preflight",
        action="store_true",
        default=defaults.get("receiver_compute_preflight", False),
    )
    parser.add_argument(
        "--run_label",
        default=defaults.get("run_label"),
        help="Label this preflight variant in run names and trace tags.",
    )
    parser.add_argument(
        "--receiver_budgets",
        default=defaults.get("receiver_budgets", "64,128,256,512,1024,free"),
    )
    parser.add_argument(
        "--target_accuracies", default=defaults.get("target_accuracies", "0.5,0.7,0.9")
    )
    parser.add_argument(
        "--bootstrap_count", type=int, default=defaults.get("bootstrap_count", 2000)
    )
    parser.add_argument(
        "--free_max_new_tokens",
        type=int,
        default=defaults.get("free_max_new_tokens", 8192),
        help="Retry cap-limited free trajectories up to this ceiling; unresolved cases are pathological.",
    )
    parser.add_argument(
        "--verify_prefix",
        action="store_true",
        default=defaults.get("verify_prefix", False),
    )


def validate_compute_preflight_args(parser, args):
    if not args.receiver_compute_preflight:
        return
    if any((args.receiver_reasoning, args.intervention, args.acquisition)):
        parser.error("experiment use cases are mutually exclusive")
    if args.task != "gsm8k" or args.method != "latent_mas" or args.use_vllm:
        parser.error("preflight requires GSM8K transformers LatentMAS")
    if (
        args.handoff_positions is not None
        or getattr(args, "latent_only", False)
        or getattr(args, "sequential_info_only", False)
    ):
        parser.error("preflight requires full KV without truncation")
    try:
        args.upstream_steps = [int(x) for x in str(args.upstream_steps).split(",")]
        args.receiver_budgets = [
            "free" if x == "free" else int(x)
            for x in str(args.receiver_budgets).split(",")
        ]
        args.target_accuracies = [
            float(x) for x in str(args.target_accuracies).split(",")
        ]
    except ValueError:
        parser.error("invalid comma-separated preflight axis")
    if (
        not args.upstream_steps
        or args.upstream_steps != sorted(set(args.upstream_steps))
        or min(args.upstream_steps) <= 0
    ):
        parser.error("upstream_steps must be positive increasing unique integers")
    finite = [b for b in args.receiver_budgets if b != "free"]
    if (
        not finite
        or finite != sorted(set(finite))
        or min(finite) <= 0
        or args.receiver_budgets != [*finite, "free"]
    ):
        parser.error(
            "receiver_budgets requires increasing positive budgets followed by free"
        )
    if args.max_new_tokens <= max(finite) or args.bootstrap_count < 1:
        parser.error(
            "free cap must exceed finite budgets; bootstrap_count must be positive"
        )
    if args.free_max_new_tokens < args.max_new_tokens:
        parser.error("free_max_new_tokens must be at least max_new_tokens")
    if not args.target_accuracies or any(
        not 0 <= x <= 1 for x in args.target_accuracies
    ):
        parser.error("target accuracies must be between zero and one")
    if args.max_samples != -1 and args.max_samples < 2:
        parser.error("preflight requires at least two samples")
    if args.verify_prefix and args.max_samples != 2:
        parser.error("prefix verification is restricted to a two-sample smoke")
    args.temperature, args.top_p = 0.0, 1.0
