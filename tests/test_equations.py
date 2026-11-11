import pytest
from backend.app.engine.powder_factor import powder_factor_mass
from backend.app.engine.fragmentation import rosin_rammler_passing_fraction
from backend.app.engine.vibration import scaled_distance_ppv

def test_powder_factor():
    assert powder_factor_mass(100, 1000) == pytest.approx(0.1)

def test_rosin_rammler_zero_size():
    assert rosin_rammler_passing_fraction(0, 50, 1.5) == pytest.approx(0.0)

def test_ppv_requires_positive_inputs():
    with pytest.raises(ValueError):
        scaled_distance_ppv(0, 10, 100, 1.5)
