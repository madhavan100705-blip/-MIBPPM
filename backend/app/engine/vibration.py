def scaled_distance_ppv(distance_m: float, max_charge_per_delay_kg: float,
                        K: float, beta: float) -> float:
    """Site-calibrated scaled-distance PPV equation.

    No default K or beta is provided. These constants must be calibrated from
    measured vibration records for the relevant site/domain.
    """
    if min(distance_m, max_charge_per_delay_kg, K, beta) <= 0:
        raise ValueError("distance, charge, K and beta must be positive.")
    scaled_distance = distance_m / (max_charge_per_delay_kg ** 0.5)
    return K * (scaled_distance ** (-beta))
