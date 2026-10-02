from collections import Counter

import data_loader
from model.modules.features import OrderedRay2, get_available_features


def _features(fen: str):
    batch = data_loader.get_sparse_batch_from_fens(
        "OrderedRay2", [fen], [0], [0], [0]
    )
    try:
        raw = batch.contents
        white = [raw.white[i] for i in range(raw.max_active_features)]
        black = [raw.black[i] for i in range(raw.max_active_features)]
        return (
            [index for index in white if index != -1],
            [index for index in black if index != -1],
        )
    finally:
        data_loader.destroy_sparse_batch(batch)


def _index(source_color, source_direction, blocker, target, first_distance, second_distance):
    def bucket(distance):
        return 0 if distance == 1 else 1 if distance <= 3 else 2

    index = source_color
    for radix, value in (
        (16, source_direction),
        (12, blocker),
        (12, target),
        (3, bucket(first_distance)),
        (3, bucket(second_distance)),
    ):
        index = index * radix + value
    return index


def test_feature_is_registered_with_collision_free_size():
    assert "OrderedRay2" in get_available_features()
    assert OrderedRay2.NUM_INPUTS == 41_472
    assert OrderedRay2.MAX_ACTIVE_FEATURES == 240


def test_extracts_ordered_rook_ray_for_both_perspectives():
    # White rook a1, white knight a3, black queen a8. The kings choose the
    # baseline orientation without a horizontal flip.
    fen = "q6k/8/8/8/8/N7/8/R3K3 w - - 0 1"
    white, black = _features(fen)

    # The relationship is directed, so the queen looking back through the
    # knight to the rook emits a second descriptor.
    assert white == [
        _index(0, 6, 1, 10, 2, 5),
        _index(1, 15, 1, 3, 5, 2),
    ]
    assert black == [
        _index(1, 7, 7, 4, 2, 5),
        _index(0, 14, 7, 9, 5, 2),
    ]


def test_preserves_feature_multiplicity():
    # Translation-invariant compact keys intentionally collide when two board
    # rays have identical descriptors; both contributions must be retained.
    fen = "q3k2q/8/8/8/8/N6N/8/R3K2R w - - 0 1"
    white, _ = _features(fen)
    expected = _index(0, 6, 1, 10, 2, 5)

    assert Counter(white)[expected] == 2


def test_requires_two_occupied_squares_beyond_slider():
    white, black = _features("7k/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert white == []
    assert black == []
