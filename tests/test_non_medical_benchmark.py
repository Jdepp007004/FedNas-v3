"""Fast contract tests for the isolated non-medical benchmark."""

import numpy as np


def test_digits_bundle_and_partitions_are_deterministic():
    from non_medical_benchmark.data import load_dataset, make_client_partitions

    bundle = load_dataset("digits")
    capped = load_dataset("digits", max_samples=600)
    first = make_client_partitions(bundle.y_train, num_clients=4, alpha=0.5, seed=7)
    second = make_client_partitions(bundle.y_train, num_clients=4, alpha=0.5, seed=7)
    assert bundle.input_dim == 64
    assert len(capped.X_train) == 600
    assert len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test) == 1797
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert sum(len(part) for part in first) == len(bundle.y_train)


def test_shape_safe_subnet_aggregation_preserves_uncovered_coordinates():
    from non_medical_benchmark.federated import aggregate_subnet_updates

    base = {"layer.weight": np.zeros((4, 4), dtype=np.float32)}
    updates = [
        {"layer.weight": np.ones((2, 2), dtype=np.float32)},
        {"layer.weight": np.full((4, 4), 3.0, dtype=np.float32)},
    ]
    merged = aggregate_subnet_updates(base, updates, [1, 1])
    np.testing.assert_allclose(merged["layer.weight"][:2, :2], 2.0)
    np.testing.assert_allclose(merged["layer.weight"][2:, 2:], 3.0)


def test_production_partial_aggregator_preserves_full_global_shape():
    from server.aggregation import aggregate_partial_fedavg

    base = {"layer.weight": np.zeros((4, 4), dtype=np.float32)}
    merged = aggregate_partial_fedavg(
        base,
        [{"layer.weight": np.ones((2, 2), dtype=np.float32)},
         {"layer.weight": np.full((4, 4), 3.0, dtype=np.float32)}],
        [1, 1],
    )
    assert merged["layer.weight"].shape == (4, 4)
    np.testing.assert_allclose(merged["layer.weight"][2:, 2:], 3.0)


def test_elastic_model_forward_and_subnet_export():
    import torch
    from non_medical_benchmark.models import (
        ElasticMLP,
        build_fixed_subnet,
        extract_subnet_state,
        load_subnet_state,
    )

    model = ElasticMLP(input_dim=8, num_classes=3, max_depth=3, hidden_dim=6)
    logits = model(torch.randn(5, 8), active_depth=2, active_width=4)
    assert logits.shape == (5, 3)
    update = extract_subnet_state(model, depth=2, width=4)
    assert update["layers.0.weight"].shape == (4, 8)
    assert update["layers.1.weight"].shape == (4, 4)
    assert update["classifier.weight"].shape == (3, 4)
    assert "layers.2.weight" not in update

    fixed = build_fixed_subnet("elastic", 8, 3, depth=2, width=4)
    load_subnet_state(fixed, {key: value.detach().numpy() for key, value in model.state_dict().items()})
    assert sum(parameter.numel() for parameter in fixed.parameters()) < sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert fixed(torch.randn(5, 8)).shape == (5, 3)


def test_one_round_elastic_smoke_run():
    from non_medical_benchmark.config import get_dataset_spec
    from non_medical_benchmark.data import load_dataset
    from non_medical_benchmark.federated import run_federated_experiment

    spec = get_dataset_spec("digits")
    bundle = load_dataset("digits")
    result = run_federated_experiment(bundle, spec, variant="elastic", rounds=1, num_clients=3)
    assert result["final"]["selected_val_accuracy"] >= 0.0
    assert result["final"]["selected_test_accuracy"] >= 0.0
    assert result["final"]["mean_upload_bytes"] > 0
    assert result["final"]["mean_client_parameters"] > 0


def test_strategy_and_scaffold_paths_are_reproducible():
    from non_medical_benchmark.config import get_dataset_spec
    from non_medical_benchmark.data import load_dataset
    from non_medical_benchmark.federated import run_federated_experiment

    spec = get_dataset_spec("digits")
    bundle = load_dataset("digits", max_samples=300)
    first = run_federated_experiment(
        bundle, spec, variant="elastic", strategy="elastic_scaffold", rounds=1,
        seed=11, num_clients=3, dropout_rate=0.2, straggler_rate=0.2,
    )
    second = run_federated_experiment(
        bundle, spec, variant="elastic", strategy="elastic_scaffold", rounds=1,
        seed=11, num_clients=3, dropout_rate=0.2, straggler_rate=0.2,
    )
    assert first["final"] == second["final"]
    assert first["final"]["drift_correction"] is True
    assert first["final"]["participating_clients"] >= 1


def test_secure_sum_masks_cancel_without_exposing_plaintext_payloads():
    from non_medical_benchmark.security import PairwiseSecureAggregator

    aggregator = PairwiseSecureAggregator([0, 1, 2], vector_dim=4, round_seed=9)
    vectors = {
        0: np.array([1.0, 2.0, 3.0, 4.0]),
        1: np.array([5.0, 6.0, 7.0, 8.0]),
        2: np.array([9.0, 10.0, 11.0, 12.0]),
    }
    masked = {client_id: aggregator.mask(client_id, value) for client_id, value in vectors.items()}
    assert not np.array_equal(masked[0], vectors[0])
    np.testing.assert_allclose(aggregator.aggregate(masked), np.sum(list(vectors.values()), axis=0))
    assert aggregator.report().key_exchange_bytes == 3 * 32


def test_trimmed_mean_rejects_a_single_coordinate_outlier():
    from non_medical_benchmark.federated import aggregate_subnet_updates

    base = {"layer.weight": np.zeros((2,), dtype=np.float32)}
    updates = [
        {"layer.weight": np.array([1.0, 1.0], dtype=np.float32)},
        {"layer.weight": np.array([1.1, 1.1], dtype=np.float32)},
        {"layer.weight": np.array([0.9, 0.9], dtype=np.float32)},
        {"layer.weight": np.array([100.0, 100.0], dtype=np.float32)},
    ]
    merged = aggregate_subnet_updates(base, updates, [1, 1, 1, 1], defense="trimmed_mean", trim_ratio=0.25)
    np.testing.assert_allclose(merged["layer.weight"], np.array([1.05, 1.05]), atol=0.01)


def test_cifar_prefix_model_and_fixed_subnet_contract():
    import torch
    from non_medical_benchmark.cifar import (
        CifarElasticCNN,
        build_cifar_subnet,
        cifar_parameters,
        extract_cifar_subnet_state,
    )
    from non_medical_benchmark.models import load_subnet_state

    model = CifarElasticCNN(num_classes=10, max_depth=3, hidden_dim=32)
    logits = model(torch.randn(2, 3, 32, 32), active_depth=2, active_width=16)
    assert logits.shape == (2, 10)
    fixed = build_cifar_subnet(10, depth=2, width=16)
    load_subnet_state(fixed, {key: value.detach().numpy() for key, value in model.state_dict().items()})
    update = extract_cifar_subnet_state(model, depth=2, width=16)
    assert update["layers.0.weight"].shape == (16, 3, 3, 3)
    assert update["layers.1.weight"].shape == (16, 16, 3, 3)
    assert sum(parameter.numel() for parameter in fixed.parameters()) == cifar_parameters(10, 2, 16)
    assert fixed(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_fedrolex_rolling_window_places_updates_at_the_declared_offset():
    from non_medical_benchmark.federated import ClientArchitecture, aggregate_subnet_updates

    base = {"layers.1.weight": np.zeros((8, 8), dtype=np.float32)}
    architecture = ClientArchitecture(depth=2, width=3, offset=4)
    update = {"layers.1.weight": np.full((3, 3), 7.0, dtype=np.float32)}
    merged = aggregate_subnet_updates(
        base, [update], [1], architectures=[architecture]
    )
    np.testing.assert_allclose(merged["layers.1.weight"][4:7, 4:7], 7.0)
    np.testing.assert_allclose(merged["layers.1.weight"][:4], 0.0)


def test_confidence_intervals_are_present_and_paired():
    from non_medical_benchmark.stats import confidence_interval, paired_interval

    interval = confidence_interval([0.8, 0.9, 1.0])
    assert interval["ci95_low"] < interval["mean"] < interval["ci95_high"]
    paired = paired_interval([0.8, 0.8, 0.8], [0.9, 0.8, 1.0])
    np.testing.assert_allclose(paired["mean"], np.mean([0.1, 0.0, 0.2]))
