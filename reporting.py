"""Run statistics collection and CSV report export for the farm simulation."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, Union

from farm import Position
from plant_class import PlantGrowthStages
from simulation import FarmSimulation
from simulation_clock import SimulationClock

ActionLogRow = TypedDict(
    "ActionLogRow",
    {
        "Day": int,
        "Time": str,
        "Elapsed Second": int,
        "Position": str,
        "Species": str,
        "Action": str,
    },
)

SpeciesSummaryRow = TypedDict(
    "SpeciesSummaryRow",
    {
        "Species": str,
        "Initial Plants": int,
        "Harvested": int,
        "Dead Plants": int,
        "Dead Removed": int,
        "Harvest Efficiency Percentage": float,
        "Planted": int,
        "pH Adjusted": int,
        "Nutrients Replenished": int,
    },
)


@dataclass(frozen=True)
class DailyCumulativeSnapshot:
    """Cumulative agent metrics captured at the end of one simulated day."""

    day: int
    completed_tasks: int
    completed_actions: int
    distance_traveled: int


class RunStatisticsRecorder:
    """Observe the actual simulation and retain daily cumulative metric snapshots."""

    # Record initial species counts before the run starts.
    def __init__(self, simulation: FarmSimulation) -> None:
        """Create a recorder attached conceptually to one simulation run."""
        self.initial_species_counts = simulation.farm.species_counts().copy()
        self.initial_dead_species_by_position: dict[Position, str] = {
            position: simulation.farm.get_plant(position).species.name
            for position in simulation.farm.positions()
            if simulation.farm.get_plant(position).current_growth_stage
               == PlantGrowthStages.DEAD
        }
        self.daily_snapshots: list[DailyCumulativeSnapshot] = []

    # Capture one cumulative snapshot whenever the authoritative clock rolls to a new day.
    def observe(self, simulation: FarmSimulation) -> None:
        """Record end-of-day cumulative metrics from the actual simulation."""
        if simulation.clock.second_of_day != 0:
            return
        day = simulation.clock.current_day
        if day <= 0:
            return
        if self.daily_snapshots and self.daily_snapshots[-1].day == day:
            return

        self.daily_snapshots.append(
            DailyCumulativeSnapshot(
                day=day,
                completed_tasks=simulation.agent.completed_tasks,
                completed_actions=simulation.agent.completed_actions,
                distance_traveled=simulation.agent.distance_traveled,
            )
        )

    # Convert the action record list into detailed rows with day-local timestamps.
    @staticmethod
    def action_log_rows(simulation: FarmSimulation) -> list[ActionLogRow]:
        """Return one report row for every completed service action."""
        rows: list[ActionLogRow] = []
        for record in simulation.agent.action_records:
            elapsed = record["elapsed_second"]
            day = (elapsed - 1) // SimulationClock.SECONDS_PER_DAY + 1
            second_of_day = elapsed % SimulationClock.SECONDS_PER_DAY
            if elapsed > 0 and second_of_day == 0:
                time_text = "24:00:00"
            else:
                hours = second_of_day // 3600
                minutes = (second_of_day % 3600) // 60
                seconds = second_of_day % 60
                time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            rows.append(
                {
                    "Day": day,
                    "Time": time_text,
                    "Elapsed Second": elapsed,
                    "Position": str(record["position"]),
                    "Species": record["species"],
                    "Action": record["action"],
                }
            )
        return rows

    # Aggregate completed service actions into one row per simulated day.
    def daily_summary_rows(self, simulation: FarmSimulation) -> list[dict[str, int]]:
        """Return per-day action counts plus task and distance deltas."""
        action_counts: dict[int, dict[str, int]] = {}
        for row in self.action_log_rows(simulation):
            day = row["Day"]
            action = row["Action"]
            day_counts = action_counts.setdefault(day, {})
            day_counts[action] = day_counts.get(action, 0) + 1

        rows: list[dict[str, int]] = []
        previous_tasks = 0
        previous_actions = 0
        previous_distance = 0

        for snapshot in self.daily_snapshots:
            counts = action_counts.get(snapshot.day, {})
            rows.append(
                {
                    "Day": snapshot.day,
                    "Harvested": counts.get("HARVEST", 0),
                    "Planted": counts.get("PLANT", 0),
                    "pH Adjusted": counts.get("ADJUST_PH", 0),
                    "Nutrients Replenished": counts.get("REPLENISH_NUTRIENTS", 0),
                    "Dead Removed": counts.get("REMOVE_DEAD", 0),
                    "Tasks Completed": snapshot.completed_tasks - previous_tasks,
                    "Actions Completed": snapshot.completed_actions - previous_actions,
                    "Distance Traveled": snapshot.distance_traveled - previous_distance,
                }
            )
            previous_tasks = snapshot.completed_tasks
            previous_actions = snapshot.completed_actions
            previous_distance = snapshot.distance_traveled

        return rows

    # Aggregate actions and harvest efficiency for each species across the full run.
    def species_summary_rows(
            self,
            simulation: FarmSimulation,
    ) -> list[SpeciesSummaryRow]:
        """Return action totals and harvest-versus-death efficiency by species."""
        summaries: dict[str, dict[str, int]] = {}
        for species in self.initial_species_counts:
            summaries[species] = {
                "Harvested": 0,
                "Planted": 0,
                "Dead Removed": 0,
                "pH Adjusted": 0,
                "Nutrients Replenished": 0,
            }

        action_to_column = {
            "HARVEST": "Harvested",
            "PLANT": "Planted",
            "REMOVE_DEAD": "Dead Removed",
            "ADJUST_PH": "pH Adjusted",
            "REPLENISH_NUTRIENTS": "Nutrients Replenished",
        }

        for record in simulation.agent.action_records:
            species = record["species"]
            if species not in summaries:
                summaries[species] = {
                    "Harvested": 0,
                    "Planted": 0,
                    "Dead Removed": 0,
                    "pH Adjusted": 0,
                    "Nutrients Replenished": 0,
                }
            column = action_to_column.get(record["action"])
            if column is not None:
                summaries[species][column] += 1

        rows: list[SpeciesSummaryRow] = []
        dead_counts = {name: 0 for name in summaries}
        for species_name in self.initial_dead_species_by_position.values():
            dead_counts[species_name] = dead_counts.get(species_name, 0) + 1

        unmatched_initial_dead = set(self.initial_dead_species_by_position)
        for record in simulation.agent.action_records:
            if record["action"] != "REMOVE_DEAD":
                continue
            position = record["position"]
            if position in unmatched_initial_dead:
                unmatched_initial_dead.remove(position)
            else:
                species_name = record["species"]
                dead_counts[species_name] = dead_counts.get(species_name, 0) + 1

        for name in sorted(summaries):
            initial_plants = self.initial_species_counts.get(name, 0)
            harvested = summaries[name]["Harvested"]
            dead_removed = summaries[name]["Dead Removed"]
            dead_plants = dead_counts.get(name, 0)
            completed_outcomes = harvested + dead_plants
            harvest_efficiency = (
                harvested / completed_outcomes * 100.0
                if completed_outcomes > 0
                else 0.0
            )
            rows.append(
                {
                    "Species": name,
                    "Initial Plants": initial_plants,
                    "Harvested": harvested,
                    "Dead Plants": dead_plants,
                    "Dead Removed": dead_removed,
                    "Harvest Efficiency Percentage": round(
                        harvest_efficiency,
                        2,
                    ),
                    "Planted": summaries[name]["Planted"],
                    "pH Adjusted": summaries[name]["pH Adjusted"],
                    "Nutrients Replenished": summaries[name][
                        "Nutrients Replenished"
                    ],
                }
            )
        return rows

    # Build overall key performance indicators for the complete run.
    @staticmethod
    def run_summary_rows(simulation: FarmSimulation) -> list[tuple[str, object]]:
        """Return overall simulation metrics as metric-value pairs."""
        counts: dict[str, int] = {}
        for record in simulation.agent.action_records:
            action = record["action"]
            counts[action] = counts.get(action, 0) + 1

        return [
            ("Simulation Days", simulation.clock.current_day),
            ("Total Plants", len(list(simulation.farm.positions()))),
            ("Distance Traveled", simulation.agent.distance_traveled),
            ("Tasks Completed", simulation.agent.completed_tasks),
            ("Actions Completed", simulation.agent.completed_actions),
            ("Harvests", counts.get("HARVEST", 0)),
            ("Plants Planted", counts.get("PLANT", 0)),
            ("Dead Plants Removed", counts.get("REMOVE_DEAD", 0)),
            ("pH Adjustments", counts.get("ADJUST_PH", 0)),
            ("Nutrient Replenishments", counts.get("REPLENISH_NUTRIENTS", 0)),
        ]

    # Write the structured report tables to CSV files for portable downstream analysis.
    def write_csv_reports(
            self,
            simulation: FarmSimulation,
            output_directory: Union[str, Path],
    ) -> list[Path]:
        """Write run summary, daily summary, species summary, and action log CSV files."""
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        datasets: dict[str, Sequence[Mapping[str, object]]] = {
            "run_summary.csv": [
                {"Metric": metric, "Value": value}
                for metric, value in self.run_summary_rows(simulation)
            ],
            "daily_summary.csv": self.daily_summary_rows(simulation),
            "species_summary.csv": self.species_summary_rows(simulation),
            "action_log.csv": self.action_log_rows(simulation),
        }

        for filename, rows in datasets.items():
            path = output_directory / filename
            if rows:
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                path.write_text("", encoding="utf-8")
            paths.append(path)

        return paths
