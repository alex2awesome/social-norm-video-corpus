import pytest

from scripts.score_instructional_v20_shadow_population import (
    LOW_LEVEL,
    require_fresh_outputs,
    vector,
)


def test_vector_has_frozen_low_level_order() -> None:
    row = {
        "low_level": {
            name: float(index) for index, name in enumerate(LOW_LEVEL)
        }
    }
    assert vector(row) == [float(index) for index in range(len(LOW_LEVEL))]


def test_expansion_refuses_to_replace_frozen_output(tmp_path) -> None:
    output = tmp_path / "ranking.jsonl"
    summary = tmp_path / "summary.json"
    output.write_text("already frozen\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        require_fresh_outputs(output, summary)
