import math

def kuznesov_mean_fragment_size_cm(rock_factor: float, rock_volume_m3: float,
                                   explosive_mass_kg: float, relative_weight_strength: float) -> float:
    """Calculate Kuznetsov-style mean fragment size using supplied equation inputs.

    IMPORTANT: Verify the exact equation form, unit conventions, and rock-factor
    definition against the original source before research use.
    """
    if rock_factor <= 0 or rock_volume_m3 <= 0 or explosive_mass_kg <= 0 or relative_weight_strength <= 0:
        raise ValueError("All inputs must be positive.")
    return rock_factor * (rock_volume_m3 / explosive_mass_kg) ** 0.8 * explosive_mass_kg ** (1/6) * (115 / relative_weight_strength) ** (19/30)

def rosin_rammler_passing_fraction(size: float, characteristic_size: float, uniformity_index: float) -> float:
    """Fraction passing at size x; returns a value between 0 and 1."""
    if size < 0 or characteristic_size <= 0 or uniformity_index <= 0:
        raise ValueError("size must be >= 0; characteristic size and index must be positive.")
    return 1 - math.exp(-((size / characteristic_size) ** uniformity_index))
