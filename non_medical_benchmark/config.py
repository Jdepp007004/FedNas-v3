"""Dataset-specific configurations for the non-medical benchmark track.

The configs deliberately keep the same broad idea as the production project
(an elastic MLP whose prefixes are trainable on clients with different
budgets), while tuning width, depth, batch size, and learning rate per task.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    source_url: str
    task: str
    num_classes: int
    max_samples: Optional[int]
    max_depth: int
    hidden_dim: int
    widths: Tuple[int, ...]
    clients: int
    dirichlet_alpha: float
    batch_size: int
    local_epochs: int
    learning_rate: float
    fedprox_mu: float
    selection_metric: str = "val_accuracy"
    class_weighting: str = "none"


DATASETS = {
    "digits": DatasetSpec(
        name="digits",
        description="8x8 handwritten-digit images; offline smoke benchmark.",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        task="multiclass classification",
        num_classes=10,
        max_samples=None,
        max_depth=4,
        hidden_dim=64,
        widths=(16, 32, 48, 64),
        clients=6,
        dirichlet_alpha=0.5,
        batch_size=32,
        local_epochs=2,
        learning_rate=1e-3,
        fedprox_mu=1e-3,
        selection_metric="val_accuracy",
        class_weighting="none",
    ),
    "adult": DatasetSpec(
        name="adult",
        description="UCI Adult income classification; mixed categorical/numeric tabular data.",
        source_url="https://archive.ics.uci.edu/dataset/2/adult",
        task="binary classification",
        num_classes=2,
        max_samples=12000,
        max_depth=4,
        hidden_dim=128,
        widths=(32, 64, 96, 128),
        clients=8,
        dirichlet_alpha=0.3,
        batch_size=128,
        local_epochs=1,
        learning_rate=1e-3,
        fedprox_mu=5e-3,
        selection_metric="val_macro_f1",
        class_weighting="balanced",
    ),
    "bank_marketing": DatasetSpec(
        name="bank_marketing",
        description="UCI Bank Marketing subscription prediction; mixed tabular business data.",
        source_url="https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing",
        task="binary classification",
        num_classes=2,
        max_samples=12000,
        max_depth=4,
        hidden_dim=128,
        widths=(32, 64, 96, 128),
        clients=8,
        dirichlet_alpha=0.3,
        batch_size=128,
        local_epochs=1,
        learning_rate=1e-3,
        fedprox_mu=5e-3,
        selection_metric="val_macro_f1",
        class_weighting="balanced",
    ),
    "har": DatasetSpec(
        name="har",
        description="UCI Human Activity Recognition; 561-feature smartphone sensor data.",
        source_url="https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones",
        task="multiclass classification",
        num_classes=6,
        max_samples=12000,
        max_depth=5,
        hidden_dim=96,
        widths=(24, 48, 72, 96),
        clients=8,
        dirichlet_alpha=0.5,
        batch_size=128,
        local_epochs=1,
        learning_rate=1e-3,
        fedprox_mu=5e-3,
        selection_metric="val_macro_f1",
        class_weighting="balanced",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a named dataset configuration with a useful error message."""
    try:
        return DATASETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}") from exc
