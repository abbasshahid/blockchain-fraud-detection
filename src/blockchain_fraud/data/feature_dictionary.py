from __future__ import annotations


def feature_names(num_features: int) -> list[str]:
    return [f"feature_{i:03d}" for i in range(num_features)]


def feature_groups(num_features: int, local_feature_count: int = 93) -> dict[str, list[int]]:
    local_end = min(local_feature_count, num_features)
    return {
        "local": list(range(local_end)),
        "aggregated": list(range(local_end, num_features)),
    }

