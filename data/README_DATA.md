# Public blast-event dataset dictionary

One row should represent one blast event or one clearly identified measurement event.
Do not merge hole-level records with blast-level records without a documented aggregation method.

## Required provenance columns
- `blast_id`: unique blast-event key within the project
- `mine_id`: mine/site identifier (anonymize if required by license)
- `source_id`: key into your source register
- `source_url`: publication/repository URL where permitted
- `event_date`: date or year if available

## Suggested input columns (use blanks when unknown)
- `rock_type`, `UCS_MPa`, `BTS_MPa`, `youngs_modulus_GPa`, `poisson_ratio`
- `rock_density_kg_m3`, `RQD_pct`, `GSI`, `RMR`, `joint_spacing_m`
- `explosive_type`, `explosive_density_kg_m3`, `VOD_m_s`, `RWS_pct`
- `explosive_mass_kg`, `charge_per_delay_kg`
- `burden_m`, `spacing_m`, `bench_height_m`, `hole_diameter_mm`
- `hole_depth_m`, `stemming_m`, `subdrilling_m`, `hole_deviation_m`
- `monitor_distance_m`, `wind_speed_m_s`, `air_pressure_kPa`

## Measured output columns
- `X50_actual`, `X80_actual`, `PPV_actual`
- `flyrock_actual`, `backbreak_actual`, `air_overpressure_actual`
- `blasted_rock_mass_t`, `blasted_rock_volume_m3`
- `cost_actual`, `productivity_actual`

## Rules
1. Store units in column names or a separate data dictionary.
2. Do not substitute a model-generated value for an actual measured value.
3. Record measurement method and instrument where available.
4. Keep source citation and license details in a separate source register.
5. Do not assume a blank value is zero.
6. Preserve raw source files unchanged.
