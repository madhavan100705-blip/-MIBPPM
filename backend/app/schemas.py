from pydantic import BaseModel, Field

class FragmentationRequest(BaseModel):
    rock_factor: float = Field(gt=0)
    rock_volume_m3: float = Field(gt=0)
    explosive_mass_kg: float = Field(gt=0)
    relative_weight_strength: float = Field(gt=0)

class PPVRequest(BaseModel):
    distance_m: float = Field(gt=0)
    max_charge_per_delay_kg: float = Field(gt=0)
    K: float = Field(gt=0, description="Site-calibrated constant; no default")
    beta: float = Field(gt=0, description="Site-calibrated exponent; no default")

class PowderFactorRequest(BaseModel):
    explosive_mass_kg: float = Field(ge=0)
    rock_mass_t: float | None = Field(default=None, gt=0)
    rock_volume_m3: float | None = Field(default=None, gt=0)
