"""Select dataset samples without changing their original identities."""

from itertools import islice


def select_samples(dataset, args):
    indexed = enumerate(dataset)
    indices = getattr(args, "sample_indices", None)
    if indices is not None:
        requested = set(indices)
        if not requested or any(index < 0 for index in requested):
            raise ValueError("sample_indices must contain non-negative dataset indices")
        selected = [
            (index, item)
            for index, item in islice(indexed, max(requested) + 1)
            if index in requested
        ]
        missing = requested - {index for index, _ in selected}
        if missing:
            raise ValueError(f"Sample indices outside the dataset: {sorted(missing)}")
        return selected
    return islice(indexed, args.max_samples) if args.max_samples > 0 else indexed
