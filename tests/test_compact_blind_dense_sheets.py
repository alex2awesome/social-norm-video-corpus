import numpy as np

from scripts.compact_blind_dense_sheets import compact_panels


def test_compact_panels_preserves_order_without_pixel_loss() -> None:
    source = np.vstack(
        [np.full((2, 3, 1), value, dtype=np.uint8) for value in range(8)]
    )
    compact = compact_panels(source, panel_count=8, columns=2)

    assert compact.shape == (8, 6, 1)
    assert [
        (int(compact[row * 2, 0, 0]), int(compact[row * 2, 3, 0]))
        for row in range(4)
    ] == [(0, 1), (2, 3), (4, 5), (6, 7)]
