"""Lookup and scalarization for the downloaded Olympus LNP3 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DEFAULT_CSV_PATH = Path(__file__).with_name("lnp3.csv")
DRUG_INPUT_LEVELS = [6.0, 12.0, 24.0, 48.0]
SOLID_LIPID_INPUT_LEVELS = [72.0, 96.0, 108.0, 120.0]
LIQUID_LIPID_INPUT_LEVELS = [0.0, 12.0, 24.0, 48.0]
SURFACTANT_INPUT_LEVELS = [0.0, 0.0025, 0.005, 0.01]


def _snap(value: float, levels: Sequence[float]) -> float:
    return min(levels, key=lambda level: abs(float(value) - level))


def decode_solid_lipid(bit0: int, bit1: int) -> str:
    if int(bit0) == 0:
        return "Stearic_acid"
    return "Compritol_888" if int(bit1) == 0 else "Glyceryl_monostearate"


class LNP3Evaluator:
    """Snap to the discrete design and return three targets plus total_score."""

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        *,
        w_dl: float = 0.4,
        w_ee: float = 0.4,
        w_pd: float = 0.2,
        normalize_weights: bool = True,
    ):
        required = [
            "drug_input", "solid_lipid", "solid_lipid_input", "liquid_lipid_input",
            "surfactant_input", "drug_loading", "encap_efficiency", "particle_diameter",
        ]
        frame = pd.read_csv(csv_path)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        frame["solid_lipid"] = frame["solid_lipid"].astype(str).str.strip()
        for column in [name for name in required if name != "solid_lipid"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required).copy()
        if len(frame) != 768:
            raise ValueError(f"LNP3 CSV must contain 768 complete rows; got {len(frame)}")

        self.frame = frame.set_index(
            [
                "drug_input", "solid_lipid", "solid_lipid_input",
                "liquid_lipid_input", "surfactant_input",
            ]
        ).sort_index()
        self.minmax = {
            column: (float(frame[column].min()), float(frame[column].max()))
            for column in ["drug_loading", "encap_efficiency", "particle_diameter"]
        }
        weights = [float(w_dl), float(w_ee), float(w_pd)]
        if normalize_weights:
            total = sum(weights)
            if total <= 0:
                raise ValueError("the scalarization weights must have a positive sum")
            weights = [weight / total for weight in weights]
        self.w_dl, self.w_ee, self.w_pd = weights

    def _norm01(self, value: float, column: str) -> float:
        lower, upper = self.minmax[column]
        return 0.0 if upper <= lower else (float(value) - lower) / (upper - lower)

    def evaluate_from_dict(self, parameters: dict[str, Any]) -> dict[str, float]:
        if "solid_lipid" in parameters:
            lipid = str(parameters["solid_lipid"]).strip()
        else:
            lipid = decode_solid_lipid(
                int(parameters["solid_lipid_bit0"]),
                int(parameters.get("solid_lipid_bit1", 0)),
            )
        key = (
            _snap(float(parameters["drug_input"]), DRUG_INPUT_LEVELS),
            lipid,
            _snap(float(parameters["solid_lipid_input"]), SOLID_LIPID_INPUT_LEVELS),
            _snap(float(parameters["liquid_lipid_input"]), LIQUID_LIPID_INPUT_LEVELS),
            _snap(float(parameters["surfactant_input"]), SURFACTANT_INPUT_LEVELS),
        )
        row = self.frame.loc[key]
        loading = float(row["drug_loading"])
        efficiency = float(row["encap_efficiency"])
        diameter = float(row["particle_diameter"])
        score = (
            self.w_dl * self._norm01(loading, "drug_loading")
            + self.w_ee * self._norm01(efficiency, "encap_efficiency")
            + self.w_pd * (1.0 - self._norm01(diameter, "particle_diameter"))
        )
        return {
            "drug_loading": loading,
            "encap_efficiency": efficiency,
            "particle_diameter": diameter,
            "total_score": float(score),
        }

    def evaluate_from_list(self, point: Sequence[Any]) -> dict[str, float]:
        names = [
            "drug_input", "solid_lipid_bit0", "solid_lipid_bit1",
            "solid_lipid_input", "liquid_lipid_input", "surfactant_input",
        ]
        if len(point) != len(names):
            raise ValueError(f"expected {len(names)} values in order {names}")
        return self.evaluate_from_dict(dict(zip(names, point)))

    def evaluate(self, point: dict[str, Any] | Sequence[Any]) -> dict[str, float]:
        return self.evaluate_from_dict(point) if isinstance(point, dict) else self.evaluate_from_list(point)


class DiscreteBONormalizer:
    def __init__(self):
        self.grids = [
            DRUG_INPUT_LEVELS, [0.0, 1.0], [0.0, 1.0],
            SOLID_LIPID_INPUT_LEVELS, LIQUID_LIPID_INPUT_LEVELS,
            SURFACTANT_INPUT_LEVELS,
        ]
        self.d = len(self.grids)

    def normalize_point(self, point: Sequence[float]) -> list[float]:
        if len(point) != self.d:
            raise ValueError(f"expected {self.d} values")
        result = []
        for value, grid in zip(point, self.grids):
            snapped = _snap(float(value), grid)
            result.append(grid.index(snapped) / (len(grid) - 1))
        return result

    def denormalize_point(self, point: Sequence[float]) -> list[float]:
        if len(point) != self.d:
            raise ValueError(f"expected {self.d} values")
        result = []
        for value, grid in zip(point, self.grids):
            index = round(min(max(float(value), 0.0), 1.0) * (len(grid) - 1))
            result.append(grid[index])
        if int(result[1]) == 0:
            result[2] = 0.0
        return result


def make_bo_normalizer() -> DiscreteBONormalizer:
    return DiscreteBONormalizer()


if __name__ == "__main__":
    evaluator = LNP3Evaluator()
    point = [6, 0, 0, 120, 0, 0.0]
    print("point:", point)
    print("evaluation:", evaluator.evaluate_from_list(point))
