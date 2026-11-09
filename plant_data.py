"""CSV loading utilities for one or more plant species."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

from plant_class import PlantSpecies

REQUIRED_COLUMNS = {
    "name",
    "harvest_age",
    "max_age",
    "ph_min",
    "ph_max",
    "ec_min",
    "ec_max",
    "temperature_min",
    "temperature_max",
}


# Load all plant species definitions from a CSV file keyed by species name.
def load_plant_species_csv(
        csv_path: Union[str, Path],
) -> dict[str, PlantSpecies]:
    """Read a plant-species CSV file and return species keyed by name."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Plant data file not found: {path}")

    species_by_name: dict[str, PlantSpecies] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Plant data CSV has no header row.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Plant data CSV is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                species = PlantSpecies(
                    name=row["name"].strip(),
                    harvest_age=int(row["harvest_age"]),
                    max_age=int(row["max_age"]),
                    ph_range=(float(row["ph_min"]), float(row["ph_max"])),
                    ec_range=(float(row["ec_min"]), float(row["ec_max"])),
                    temperature_range=(
                        float(row["temperature_min"]),
                        float(row["temperature_max"]),
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid plant data on CSV row {row_number}: {exc}"
                ) from exc

            if species.name in species_by_name:
                raise ValueError(
                    f"Duplicate species name '{species.name}' on CSV row {row_number}."
                )

            species_by_name[species.name] = species

    if not species_by_name:
        raise ValueError("Plant data CSV does not contain any species.")

    return species_by_name
