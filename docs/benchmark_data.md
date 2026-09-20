# Benchmark data and preprocessing

The benchmark CSV files are downloaded from their original public sources and
are not stored in this repository. From the repository root, prepare all four:

```bash
python scripts/download_benchmarks.py
```

Or select individual datasets:

```bash
python scripts/download_benchmarks.py --datasets hplc cb lnp3 concrete
```

The default output directory is `fun/`. Use `--output-dir path/to/data` to write
elsewhere. The downloader does not use file hashes. It checks the documented
column layout and resulting row counts so that an HTML error page or a changed
source schema is not silently treated as benchmark data. It will not overwrite
a different local file.

## HPLC

Download: [Olympus HPLC data.csv](https://raw.githubusercontent.com/aspuru-guzik-group/olympus/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_hplc/data.csv)

Description: [Olympus HPLC dataset documentation](https://github.com/aspuru-guzik-group/olympus/blob/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_hplc/description.txt)

The Olympus file contains 1,386 headerless rows. LGBO prepends:

```text
sample_loop,additional_volume,tubing_volume,sample_flow,push_speed,wait_time,peak_area
```

No rows or values are otherwise changed. Olympus attributes the measurements to
Roch et al., *ChemOS: an orchestration software to democratize autonomous
discovery* (2018), [doi:10.26434/chemrxiv.5953606.v1](https://doi.org/10.26434/chemrxiv.5953606.v1),
and points to [Phoenics](https://github.com/aspuru-guzik-group/phoenics) as the
original download location.

`fun.hplc.HPLCEvaluator` maximizes `peak_area`. The six inputs are min-max scaled
using their observed bounds. The dataset is treated as scattered data because
the number of unique levels is too large for a Cartesian grid. Evaluation uses
the 12 nearest rows and inverse-distance weights
`1 / (distance + 1e-12)^2`. At an exact input match, the first matching row is
returned. Repeated HPLC input vectors are retained in their source order.

## Crossed-barrel (CB)

Download: [Olympus Crossed-barrel data.csv](https://raw.githubusercontent.com/aspuru-guzik-group/olympus/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_crossed_barrel/data.csv)

Description: [Olympus Crossed-barrel dataset documentation](https://github.com/aspuru-guzik-group/olympus/blob/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_crossed_barrel/description.txt)

The Olympus file contains 600 headerless rows. LGBO prepends:

```text
n,theta,r,t,toughness
```

No rows or values are otherwise changed. The original study is Gongora et al.,
*A Bayesian experimental autonomous researcher for mechanical design*, Science
Advances 6(15), eaaz1708 (2020),
[doi:10.1126/sciadv.aaz1708](https://doi.org/10.1126/sciadv.aaz1708).

`fun.cb.CBEvaluator` maximizes `toughness`. Its 600 observations occupy only
part of the possible Cartesian grid, so it uses the same observed-bound
min-max scaling and 12-neighbor inverse-distance interpolation described for
HPLC.

## LNP3

Download: [Olympus LNP3 data.csv](https://raw.githubusercontent.com/aspuru-guzik-group/olympus/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_lnp3/data.csv)

Description: [Olympus LNP3 dataset documentation](https://github.com/aspuru-guzik-group/olympus/blob/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_lnp3/description.txt)

The Olympus file contains 768 headerless rows. LGBO prepends:

```text
drug_input,solid_lipid,solid_lipid_input,liquid_lipid_input,surfactant_input,drug_loading,encap_efficiency,particle_diameter
```

No rows or values are otherwise changed. `surfactant_input` is LGBO's corrected
spelling of the `surfractant_input` name found in the upstream configuration.

`fun.lnp3.LNP3Evaluator` snaps each numeric input to its documented discrete
level and performs an exact row lookup. The three targets are independently
min-max normalized over the complete dataset. The scalar objective is maximized:

```text
total_score = 0.4 * drug_loading_norm
            + 0.4 * encap_efficiency_norm
            + 0.2 * (1 - particle_diameter_norm)
```

The solid lipid is represented in the BO interface by two bits: `00` maps to
`Stearic_acid`, `10` maps to `Compritol_888`, and `11` maps to
`Glyceryl_monostearate`. When the first bit is zero, the second bit is ignored.

## Concrete

Download: [UCI Concrete Compressive Strength data.csv](https://archive.ics.uci.edu/static/public/165/data.csv)

Dataset page: [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

The UCI source contains 1,030 rows, eight inputs, and compressive strength. LGBO
derives its seven-input, 968-row benchmark as follows:

1. Retain rows for which `Age <= 100`.
2. Remove the `Age` column.
3. Preserve the remaining row order and numeric values.
4. Rename the columns to:

```text
Cement,Blast,Fly,Water,Superplas,CoarseAg,FineAggr,Strength
```

The exact implementation is `prepare_concrete` in
[`scripts/download_benchmarks.py`](../scripts/download_benchmarks.py). Removing
`Age` makes this a derived seven-input benchmark rather than the unmodified UCI
task; the source contains measurements of repeated ingredient vectors at
different ages, and those retained rows remain separate observations.

`fun.concrete.ConcreteEvaluator` maximizes `Strength`. It min-max scales the
seven inputs using observed bounds, selects the 12 nearest rows, and returns an
inverse-distance-weighted value with power 2 and epsilon `1e-12`. At an exact
input match it returns the first matching row in file order.

The source is credited to I-Cheng Yeh, dataset DOI
[10.24432/C5PK67](https://doi.org/10.24432/C5PK67), and is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Running the evaluators

```bash
pip install -r requirements.txt
python scripts/download_benchmarks.py
python -m fun.hplc
python -m fun.cb
python -m fun.lnp3
python -m fun.concrete
```

Each module prints one stored-point evaluation and/or a midpoint example. The
classes also accept an explicit CSV path:

```python
from fun.hplc import HPLCEvaluator

evaluator = HPLCEvaluator("path/to/hplc.csv")
result = evaluator.evaluate_from_list([0.04, 0.03, 0.5, 1.5, 115, 5.5])
print(result["peak_area"])
```

HPLC, CB, and Concrete queries should remain inside `evaluator.bounds`. The IDW
implementation does not clip out-of-bounds queries. These modules provide the
data preparation and callable benchmark objectives; they are not wired into
the current `run_dry_once.py` command-line choices.

## Olympus attribution

HPLC, CB, and LNP3 are distributed through
[Olympus](https://github.com/aspuru-guzik-group/olympus), whose repository uses
the MIT License. The notice is retained in
[`third_party/olympus.LICENSE`](../third_party/olympus.LICENSE) and copied beside
downloaded Olympus data. Please preserve the notice and cite both Olympus and
the dataset-specific original study when redistributing or using the data.

The principal Olympus reference is Häse et al., *Olympus: a benchmarking
framework for noisy optimization and experiment planning*, Machine Learning:
Science and Technology 2, 035021 (2021),
[doi:10.1088/2632-2153/abedc8](https://doi.org/10.1088/2632-2153/abedc8).
