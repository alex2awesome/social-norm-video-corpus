import numpy as np

import scripts.benchmark_instructional_multimodal as benchmark
from scripts.benchmark_instructional_multimodal import fit_variant


def test_optional_text_model_is_loaded_lazily() -> None:
    assert "SentenceTransformer" not in vars(benchmark)


def test_fit_variant_returns_transfer_metrics() -> None:
    examples = [
        {
            "run": "v12_youtube" if index < 8 else "v15_dailymotion",
            "uid": f"u{index}",
            "audit_index": index,
        }
        for index in range(12)
    ]
    values = np.array([[index, index % 2] for index in range(12)], dtype=float)
    labels = np.array([index % 2 == 0 for index in range(12)])
    train = np.array([index < 8 for index in range(12)])
    groups = np.array([row["uid"] for row in examples])
    result = fit_variant(
        "test", values, labels, train, groups, 0.5, examples
    )
    assert result["name"] == "test"
    assert result["dailymotion_transfer_metrics"]["n"] == 4
