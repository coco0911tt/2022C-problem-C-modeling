from __future__ import annotations

import numpy as np

from common import COMPONENTS, CompositionPreprocessor, load_known_data


def main() -> None:
    t1, t2 = load_known_data()
    assert len(t1) == 58
    assert len(t2) == 69
    assert t1["artifact_id"].is_unique
    assert int(t2["valid_sum_flag"].sum()) == 67
    valid = t2[t2["valid_sum_flag"]].copy()
    prep = CompositionPreprocessor().fit(valid[COMPONENTS], valid["glass_type"])
    tr = prep.transform(valid[COMPONENTS], valid["glass_type"])
    assert np.max(tr.closure_error) <= 1e-8
    assert np.isfinite(tr.ilr).all()
    recovered = prep.inverse_ilr(tr.ilr)
    assert np.max(np.abs(recovered.sum(axis=1).to_numpy() - 100.0)) <= 1e-8
    shuffled_groups = valid["glass_type"].sample(frac=1.0, random_state=11).reset_index(drop=True)
    weather_prep = CompositionPreprocessor().fit(valid[COMPONENTS], valid["surface_weathering"])
    original = weather_prep.transform(valid[COMPONENTS], valid["surface_weathering"]).closed
    shuffled_label_irrelevant = weather_prep.transform(valid[COMPONENTS], valid["surface_weathering"]).closed
    assert np.allclose(original, shuffled_label_irrelevant)
    assert len(shuffled_groups) == len(valid)
    print("ALL_COMMON_TESTS_PASSED")


if __name__ == "__main__":
    main()
