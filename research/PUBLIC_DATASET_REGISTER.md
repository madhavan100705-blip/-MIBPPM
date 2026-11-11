# Public surface-blasting dataset register (initial research)

This register distinguishes **verified public descriptions** from **files that are actually
downloadable and licensed for reuse**. A paper describing a dataset does not automatically
make its raw rows open.

## Priority 1 — Fragmentation benchmark corpus

**CAOS_BlastFrag public repository**
- Repository: https://github.com/fsantibanezleal/CAOS_BlastFrag
- Corpus described by repository: 97 measured bench blasts from ten campaigns/sites.
- Also describes a 14-row published holdout and a separate 5-row field holdout.
- Target: measured mean fragment size (x50); predictors are dimensionless ratios and rock properties.
- Use: fragmentation model benchmarking and methodological cross-check.
- Important limitations: site-level holdout performance is much weaker than random-split performance; some sites lack enough dimensional information to reconstruct equation inputs. The repository explicitly discusses data-integrity corrections and validation leakage.
- License: repository code declares MIT. Verify the precise terms applying to each bundled data file and retain citations to the original publications before redistributing or publishing derived data.
- Status in this project: **candidate; not imported into the raw dataset yet**.

## Priority 2 — Indian limestone quarry fragmentation study

**Gebretsadik et al. (2024), “Enhancing rock fragmentation assessment in mine blasting through machine learning algorithms: a practical approach.”**
- Article: https://link.springer.com/article/10.1007/s42452-024-05888-0
- DOI: https://doi.org/10.1007/s42452-024-05888-0
- Describes 219 records from a limestone quarry in Baloda Bazaar, Chhattisgarh.
- Features include drill diameter, spacing, burden, bench height, number of holes, specific drilling, powder factor, compressive strength, stemming and charge per delay; target is fragmentation.
- Data availability statement: raw study data are available from the corresponding author on reasonable request.
- Use: potentially valuable India-specific benchmark, but it is **not currently confirmed as a directly downloadable open dataset**.
- Status in this project: **request/access required; do not fabricate or scrape approximate values from charts**.

## Priority 3 — Multi-output open-pit study

**Nobahar et al. (2024), “Advanced AI-Powered Solutions for Predicting Blast-Induced Flyrock, Backbreak, and Rock Fragmentation.”**
- Repository record: https://digital.library.adelaide.edu.au/items/7af701c9-8268-4c21-92af-17c3b8490b25
- DOI: https://doi.org/10.1007/s42461-024-01028-9
- Describes 152 compiled records from four open-pit mines in Iran.
- Targets include backbreak, flyrock and fragmentation.
- The repository record exposes the accepted manuscript, but this register does not treat the raw 152-row table as downloadable until a data file or author-provided dataset is verified.
- Use: source for target definitions, feature mapping and benchmark methodology.
- Status: **article accessible; raw table availability to verify**.

## Priority 4 — Ground vibration (PPV)

**“Assessing Ground Vibration Caused by Rock Blasting in Surface Mines Using Machine-Learning Approaches: A Comparison of CART, SVR and MARS.”**
- Article: https://www.mdpi.com/2071-1050/14/17/11060
- Describes 1,001 complete measured PPV records from 34 opencast mines, with ten blast-design variables.
- Use: strong candidate for PPV modelling and mine/site-aware validation.
- Status: **article and dataset description verified; direct raw-data download not yet verified**.

## Priority 5 — Open fragmentation observations

**El Aljibe quarry fragmentation from 3D drone image analysis**
- Dataset catalogue: https://openegiz-contest.github.io/data/
- The catalogue describes fragmentation data from 12 blasts and lists CC BY 4.0.
- Use: fragmentation distribution/image-analysis validation, subject to checking the linked Zenodo record and the file's metadata.
- Status: **open-data lead; retrieve and inspect original repository before use**.

## Priority 6 — Fragmented-rock image dataset

**Yaghoobi (2018), “fragmented rocks by blasting.”**
- Mendeley Data: https://data.mendeley.com/datasets/z78ghz96bn/1
- Describes 226 images of fragmented rocks from several rock types at Gole-Gohar iron ore mine.
- License shown by catalogue: CC BY 4.0.
- Use: image-analysis research or fragmentation measurement support; it is not by itself a structured blast-parameter-to-outcome table.
- Status: **public image dataset lead; not a substitute for numerical blast-event records**.

## Acquisition rules

1. Download only from the original repository or a verified author/institution source.
2. Preserve the source file unchanged under `data/raw/<source_id>/`.
3. Record author, title, DOI/URL, license, retrieval date, file checksum and any transformations.
4. Never reconstruct row-level measurements from figures unless the paper explicitly permits and the extraction method is documented.
5. Do not combine datasets with incompatible definitions or units without a documented harmonization step.
6. Do not merge datasets solely because they share a target label; preserve mine, site, campaign and source identifiers.
7. For every target, record the number of rows with an actual measured value. Keep missing targets missing.

## Current status

No external raw public dataset has been bundled into this prototype yet. The included
`demo_synthetic_only.csv` is explicitly synthetic and exists only to test software wiring.
The next implementation step is to acquire and audit the original source files, then create
a source-specific adapter for each compatible dataset.
