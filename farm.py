"""Three-dimensional farm environment kept separate from agent logic."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional

from plant_class import Plant, PlantRequest, PlantSpecies

Position = tuple[int, int, int]  # (floor, row, column)


@dataclass(frozen=True)
class FarmRequest:
    """A snapshot of one plant location that currently needs service."""

    position: Position
    plant: Plant
    requested_actions: tuple[PlantRequest, ...]
    priority: float


@dataclass(frozen=True)
class FarmEvent:
    """One actionable request transition produced by a plant update."""

    position: Position
    previous_actions: tuple[PlantRequest, ...]
    current_actions: tuple[PlantRequest, ...]
    priority: float


class Farm3D:
    """A rectangular 3-D farm containing exactly one plant at each coordinate."""

    MAX_PLANT_LOCATIONS = 1_000

    # Build a farm and assign one of the selected species to every plant location.
    def __init__(
            self,
            floors: int,
            rows: int,
            columns: int,
            species_options: list[PlantSpecies],
            layout_mode: str = "cycle",
            random_seed: int = 1,
    ) -> None:
        """Initialize farm dimensions and populate all plant positions."""
        if min(floors, rows, columns) <= 0:
            raise ValueError("Farm dimensions must all be greater than zero.")
        plant_locations = floors * rows * columns
        if plant_locations > self.MAX_PLANT_LOCATIONS:
            raise ValueError(
                "Farm size cannot exceed "
                f"{self.MAX_PLANT_LOCATIONS:,} plant locations."
            )
        if not species_options:
            raise ValueError("At least one plant species is required.")
        if layout_mode not in {"cycle", "random"}:
            raise ValueError("Layout mode must be 'cycle' or 'random'.")

        self.floors = floors
        self.rows = rows
        self.columns = columns
        self.species_options = list(species_options)
        self.layout_mode = layout_mode
        self.random_seed = random_seed

        layout_rng = random.Random(random_seed)
        self._plants: dict[Position, Plant] = {}

        for index, position in enumerate(self._all_positions()):
            if layout_mode == "cycle":
                species = self.species_options[index % len(self.species_options)]
            else:
                species = layout_rng.choice(self.species_options)

            self._plants[position] = Plant(
                species=species,
                rng_seed=random_seed + index + 1,
            )

    # Generate every valid coordinate in deterministic floor-row-column order.
    def _all_positions(self) -> Iterable[Position]:
        """Yield every valid farm position exactly once."""
        for floor in range(self.floors):
            for row in range(self.rows):
                for column in range(self.columns):
                    yield floor, row, column

    # Check whether a coordinate falls inside the farm dimensions.
    def in_bounds(self, position: Optional[Position]) -> bool:
        """Return True when the supplied position is inside the farm."""
        if position is None:
            return False
        floor, row, column = position
        return (
                0 <= floor < self.floors
                and 0 <= row < self.rows
                and 0 <= column < self.columns
        )

    # Return the plant object stored at a valid farm coordinate.
    def get_plant(self, position: Position) -> Plant:
        """Return the plant at a position or raise IndexError when out of bounds."""
        if not self.in_bounds(position):
            raise IndexError(f"Position {position} is outside the farm.")
        return self._plants[position]

    # Replace the plant object stored at a valid farm coordinate.
    def set_plant(self, position: Position, plant: Plant) -> None:
        """Store a specific plant object at a valid farm position."""
        if not self.in_bounds(position):
            raise IndexError(f"Position {position} is outside the farm.")
        self._plants[position] = plant

    # Expose all valid positions without exposing the farm's internal dictionary.
    def positions(self) -> Iterable[Position]:
        """Return an iterable over all occupied farm positions."""
        return self._plants.keys()

    # Advance every plant in the farm by one simulated minute.
    def simulate_minute(self) -> list[FarmEvent]:
        """Advance all plants and return only actionable request transitions."""
        events: list[FarmEvent] = []
        for position, plant in self._plants.items():
            previous_actions = tuple(plant.required_actions)
            plant.simulate_minute_for_plant()
            current_actions = tuple(plant.required_actions)
            if current_actions != previous_actions:
                events.append(
                    FarmEvent(
                        position=position,
                        previous_actions=previous_actions,
                        current_actions=current_actions,
                        priority=plant.care_priority,
                    )
                )
        return events

    # Collect a snapshot of every current plant care request for the agent.
    def collect_requests(self) -> list[FarmRequest]:
        """Return all current plant requests with positions and priorities."""
        requests: list[FarmRequest] = []

        for position, plant in self._plants.items():
            requested_actions = tuple(plant.required_actions)
            if requested_actions:
                requests.append(
                    FarmRequest(
                        position=position,
                        plant=plant,
                        requested_actions=requested_actions,
                        priority=plant.care_priority,
                    )
                )

        return requests

    # Count how many plants of each species are currently present in the farm.
    def species_counts(self) -> dict[str, int]:
        """Return current plant counts grouped by species name."""
        counts: dict[str, int] = {}
        for plant in self._plants.values():
            counts[plant.species.name] = counts.get(plant.species.name, 0) + 1
        return counts
