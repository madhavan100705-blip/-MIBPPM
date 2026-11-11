def powder_factor_mass(explosive_mass_kg: float, rock_mass_t: float) -> float:
    if explosive_mass_kg < 0 or rock_mass_t <= 0:
        raise ValueError("Explosive mass must be >= 0 and rock mass must be > 0.")
    return explosive_mass_kg / rock_mass_t

def powder_factor_volume(explosive_mass_kg: float, rock_volume_m3: float) -> float:
    if explosive_mass_kg < 0 or rock_volume_m3 <= 0:
        raise ValueError("Explosive mass must be >= 0 and rock volume must be > 0.")
    return explosive_mass_kg / rock_volume_m3
