from fastapi import FastAPI, HTTPException
from backend.app.schemas import FragmentationRequest, PPVRequest, PowderFactorRequest
from backend.app.engine.fragmentation import kuznesov_mean_fragment_size_cm, rosin_rammler_passing_fraction
from backend.app.engine.vibration import scaled_distance_ppv
from backend.app.engine.powder_factor import powder_factor_mass, powder_factor_volume

app = FastAPI(title="MIBPPM Research API", version="0.1.0",
              description="Academic research prototype; not for operational blast design.")

@app.get("/api/health")
def health():
    return {"status": "ok", "project": "MIBPPM", "mode": "research_prototype"}

@app.post("/api/mathematical/fragmentation/mean")
def fragmentation(req: FragmentationRequest):
    try:
        value = kuznesov_mean_fragment_size_cm(req.rock_factor, req.rock_volume_m3,
                                                req.explosive_mass_kg, req.relative_weight_strength)
        return {"model": "Kuznetsov-style equation", "mean_fragment_size_cm": value,
                "status": "formula_calculation_only",
                "note": "Verify original source unit conventions and rock-factor definition."}
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.post("/api/mathematical/fragmentation/passing")
def passing(size: float, characteristic_size: float, uniformity_index: float):
    try:
        return {"passing_fraction": rosin_rammler_passing_fraction(size, characteristic_size, uniformity_index)}
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.post("/api/mathematical/ppv")
def ppv(req: PPVRequest):
    try:
        return {"model": "site-calibrated scaled-distance", "PPV": scaled_distance_ppv(
            req.distance_m, req.max_charge_per_delay_kg, req.K, req.beta),
            "status": "requires_site_calibrated_constants"}
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.post("/api/mathematical/powder-factor")
def powder_factor(req: PowderFactorRequest):
    result = {}
    if req.rock_mass_t is not None:
        result["kg_per_tonne"] = powder_factor_mass(req.explosive_mass_kg, req.rock_mass_t)
    if req.rock_volume_m3 is not None:
        result["kg_per_m3"] = powder_factor_volume(req.explosive_mass_kg, req.rock_volume_m3)
    if not result:
        raise HTTPException(422, "Provide rock_mass_t and/or rock_volume_m3.")
    return result
