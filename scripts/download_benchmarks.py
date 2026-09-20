"""Download and prepare the four tabular benchmarks used by LGBO.

No checksum is required. The script performs row/column validation so that a
changed or malformed upstream response is not silently accepted as benchmark
data. See docs/benchmark_data.md for provenance and processing details.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OLYMPUS_COMMIT = "440b6b58ebfcaa2391cff7e94b570fb4fda98d68"
OLYMPUS_BASE = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/olympus/"
    f"{OLYMPUS_COMMIT}/src/olympus/datasets"
)
CONCRETE_URL = "https://archive.ics.uci.edu/static/public/165/data.csv"

OLYMPUS_DATASETS = {
    "hplc": {
        "url": f"{OLYMPUS_BASE}/dataset_hplc/data.csv",
        "header": [
            "sample_loop", "additional_volume", "tubing_volume", "sample_flow",
            "push_speed", "wait_time", "peak_area",
        ],
        "rows": 1386,
    },
    "cb": {
        "url": f"{OLYMPUS_BASE}/dataset_crossed_barrel/data.csv",
        "header": ["n", "theta", "r", "t", "toughness"],
        "rows": 600,
    },
    "lnp3": {
        "url": f"{OLYMPUS_BASE}/dataset_lnp3/data.csv",
        "header": [
            "drug_input", "solid_lipid", "solid_lipid_input",
            "liquid_lipid_input", "surfactant_input", "drug_loading",
            "encap_efficiency", "particle_diameter",
        ],
        "rows": 768,
    },
}

CONCRETE_SOURCE_COLUMNS = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
    "Coarse Aggregate", "Fine Aggregate", "Age",
    "Concrete compressive strength",
]
CONCRETE_OUTPUT_COLUMNS = [
    "Cement", "Blast", "Fly", "Water", "Superplas", "CoarseAg", "FineAggr",
    "Strength",
]
CONCRETE_SELECTED_COLUMNS = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
    "Coarse Aggregate", "Fine Aggregate", "Concrete compressive strength",
]


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "LGBO-benchmark-downloader"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def prepare_olympus(name: str, raw: bytes) -> bytes:
    """Validate dimensions and prepend the LGBO header to Olympus data."""
    spec = OLYMPUS_DATASETS[name]
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) != spec["rows"]:
        raise ValueError(f"{name}: expected {spec['rows']} rows, received {len(rows)}")
    columns = len(spec["header"])
    if any(len(row) != columns for row in rows):
        raise ValueError(f"{name}: expected {columns} columns in every row")
    header = ",".join(spec["header"]).encode("utf-8") + b"\n"
    return header + raw.lstrip(b"\xef\xbb\xbf")


def prepare_concrete(raw: bytes) -> bytes:
    """Keep Age <= 100, remove Age, and rename the remaining UCI columns."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != CONCRETE_SOURCE_COLUMNS:
        raise ValueError("concrete: upstream columns do not match the documented UCI schema")

    selected = []
    for row in reader:
        if float(row["Age"]) <= 100.0:
            selected.append([row[column] for column in CONCRETE_SELECTED_COLUMNS])
    if len(selected) != 968:
        raise ValueError(f"concrete: expected 968 rows after Age <= 100; got {len(selected)}")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CONCRETE_OUTPUT_COLUMNS)
    writer.writerows(selected)
    return output.getvalue().encode("utf-8")


def write_if_absent(path: Path, content: bytes) -> str:
    """Keep an identical file and refuse to replace a different local file."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different local file: {path}")
        return "Already present"
    path.write_bytes(content)
    return "Wrote"


def download_dataset(name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if name == "concrete":
        url = CONCRETE_URL
        prepared = prepare_concrete(fetch(url))
        notice = (ROOT / "third_party" / "concrete.SOURCE.txt").read_bytes()
        write_if_absent(output_dir / "SOURCE.concrete.txt", notice)
        row_count = 968
    else:
        spec = OLYMPUS_DATASETS[name]
        url = spec["url"]
        prepared = prepare_olympus(name, fetch(url))
        license_text = (ROOT / "third_party" / "olympus.LICENSE").read_bytes()
        write_if_absent(output_dir / "LICENSE.olympus", license_text)
        row_count = spec["rows"]

    path = output_dir / f"{name}.csv"
    status = write_if_absent(path, prepared)
    print(f"{status}: {path} ({row_count} data rows)")
    print(f"Source: {url}")
    return path


def main() -> None:
    names = [*OLYMPUS_DATASETS, "concrete"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=names, default=names,
        help="Datasets to prepare (default: all four).",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fun")
    args = parser.parse_args()
    for name in dict.fromkeys(args.datasets):
        download_dataset(name, args.output_dir)


if __name__ == "__main__":
    main()
