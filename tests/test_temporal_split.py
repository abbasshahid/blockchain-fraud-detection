from blockchain_fraud.data.temporal_split import make_temporal_split


def test_temporal_split_is_ordered_and_disjoint():
    split = make_temporal_split(range(1, 50))
    assert max(split["train_steps"]) < min(split["val_steps"])
    assert max(split["val_steps"]) < min(split["test_steps"])
    assert not set(split["train_steps"]) & set(split["test_steps"])

