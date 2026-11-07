"""Select dataset samples without changing their original identities."""

from itertools import islice


def select_samples(dataset, args):
    ids = getattr(args, "sample_ids", None)
    indices = getattr(args, "sample_indices", None)
    if ids is not None:
        if indices is not None:
            raise ValueError("sample_ids and sample_indices are mutually exclusive")
        requested = set(ids)
        if not requested or any(not sample_id for sample_id in requested):
            raise ValueError("sample_ids must contain non-empty dataset IDs")
        selected = []
        missing = requested.copy()
        for index, item in enumerate(dataset):
            sample_id = str(item["id"]) if "id" in item else None
            if sample_id in requested:
                selected.append((index, item))
                missing.discard(sample_id)
                if not missing:
                    return selected
        raise ValueError(f"Sample IDs outside the dataset: {sorted(missing)}")
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
