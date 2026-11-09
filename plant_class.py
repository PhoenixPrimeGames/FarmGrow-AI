"""Plant species definitions and per-plant simulation behavior."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

MINUTES_PER_DAY = 1440


class PlantGrowthStages(Enum):
    """Possible lifecycle states for a plant."""

    DEAD = 0
    SEED = 1
    GROWING = 2
    HARVESTABLE = 3
    HARVESTED = 4


class PlantRequest(Enum):
    """High-level care requests that a plant can expose to the farm."""

    HARVEST = "HARVEST"
    ADJUST_PH = "ADJUST_PH"
    REPLENISH_NUTRIENTS = "REPLENISH_NUTRIENTS"
    REPLACE = "REPLACE"


@dataclass(frozen=True)
class PlantSpecies:
    """Immutable data that defines the acceptable ranges and lifecycle of a species."""

    name: str
    harvest_age: int
    max_age: int
    ph_range: tuple[float, float]
    ec_range: tuple[float, float]
    temperature_range: tuple[float, float]

    # Validate a species definition immediately after dataclass construction.
    def __post_init__(self) -> None:
        """Validate species lifecycle and environmental ranges."""
        if not self.name.strip():
            raise ValueError("Species name cannot be empty.")
        if self.harvest_age <= 0:
            raise ValueError("Harvest age must be greater than zero.")
        if self.max_age <= self.harvest_age:
            raise ValueError("Maximum age must be greater than harvest age.")
        if self.ph_range[0] > self.ph_range[1]:
            raise ValueError("Invalid pH range.")
        if self.ec_range[0] > self.ec_range[1]:
            raise ValueError("Invalid EC range.")
        if self.temperature_range[0] > self.temperature_range[1]:
            raise ValueError("Invalid temperature range.")


class Plant:
    """A single simulated plant whose condition changes once per simulated minute."""

    # Create a new plant with deterministic, per-plant random drift.
    def __init__(
            self,
            species: PlantSpecies,
            rng_seed: Optional[int] = None,
    ) -> None:
        """Initialize a seed-stage plant for the supplied species."""
        self.species = species
        self.current_age_days = 0
        self.current_age_minutes = 0
        self.current_growth_stage = PlantGrowthStages.SEED
        self.plant_health_percent = 100.0
        self.current_ph_level = sum(self.species.ph_range) / 2
        self.current_ec_level = sum(self.species.ec_range) / 2
        self.rng = random.Random(rng_seed)

    # Return a concise human-readable description of the current plant state.
    def __str__(self) -> str:
        """Return a formatted summary of the plant."""
        return (
            f"{self.species.name}: "
            f"age={self.current_age_days}d {self.current_age_minutes}m, "
            f"stage={self.current_growth_stage.name}, "
            f"health={self.plant_health_percent:.2f}%, "
            f"pH={self.current_ph_level:.2f}, "
            f"EC={self.current_ec_level:.2f}"
        )

    # Advance this plant by one simulated day using the original coarse-grained model.
    def simulate_day_for_plant(self) -> bool:
        """Advance the plant by one day and return whether any state changed."""
        if self.current_growth_stage in (
                PlantGrowthStages.DEAD,
                PlantGrowthStages.HARVESTED,
        ):
            return False

        self.current_age_days += 1
        self.current_ph_level += self.rng.uniform(-0.08, 0.06)
        self.current_ec_level += self.rng.uniform(-0.05, 0.03)
        return self._update_plant()

    # Advance this plant by one simulated minute using small pH and EC drift.
    def simulate_minute_for_plant(self) -> bool:
        """Advance the plant by one minute and return whether any state changed."""
        if self.current_growth_stage in (
                PlantGrowthStages.DEAD,
                PlantGrowthStages.HARVESTED,
        ):
            return False

        self.current_age_minutes += 1
        if self.current_age_minutes >= MINUTES_PER_DAY:
            self.current_age_days += 1
            self.current_age_minutes -= MINUTES_PER_DAY

        self.current_ph_level += self.rng.uniform(-0.0002, 0.00015)
        self.current_ec_level += self.rng.uniform(-0.0001, 0.00005)
        return self._update_plant()

    # Harvest a plant only when it has reached the harvestable growth stage.
    def harvest_plant(self) -> bool:
        """Change a harvestable plant to HARVESTED and report success."""
        if self.current_growth_stage == PlantGrowthStages.HARVESTABLE:
            self.current_growth_stage = PlantGrowthStages.HARVESTED
            return True
        return False

    # Force a plant into the dead state.
    def kill_plant(self) -> None:
        """Mark the plant as dead and set its health to zero."""
        self.current_growth_stage = PlantGrowthStages.DEAD
        self.plant_health_percent = 0.0

    # Reset a harvested or dead plant location with a new plant of the supplied species.
    def replace_plant(self, species: PlantSpecies) -> bool:
        """Replace a dead or harvested plant in-place and report success."""
        if self.current_growth_stage not in (
                PlantGrowthStages.HARVESTED,
                PlantGrowthStages.DEAD,
        ):
            return False

        self.species = species
        self.current_age_days = 0
        self.current_age_minutes = 0
        self.current_growth_stage = PlantGrowthStages.SEED
        self.plant_health_percent = 100.0
        self.current_ph_level = sum(self.species.ph_range) / 2
        self.current_ec_level = sum(self.species.ec_range) / 2
        return True

    # Reset the pH level to the midpoint of the species' preferred range.
    def adjust_ph(self) -> bool:
        """Correct pH for a living, non-harvested plant and report success."""
        if self.current_growth_stage in (
                PlantGrowthStages.DEAD,
                PlantGrowthStages.HARVESTED,
        ):
            return False

        self.current_ph_level = sum(self.species.ph_range) / 2
        return True

    # Reset the EC level to the midpoint of the species' preferred range.
    def replenish_nutrients(self) -> bool:
        """Correct EC for a living, non-harvested plant and report success."""
        if self.current_growth_stage in (
                PlantGrowthStages.DEAD,
                PlantGrowthStages.HARVESTED,
        ):
            return False

        self.current_ec_level = sum(self.species.ec_range) / 2
        return True

    # Update the lifecycle stage based on the plant's current age.
    def _update_plant_stage_by_age(self) -> bool:
        """Advance lifecycle stages or kill an over-age plant."""
        if self.current_age_days > self.species.max_age:
            self.kill_plant()
            return True

        if (
                self.current_age_days >= self.species.harvest_age
                and self.current_growth_stage == PlantGrowthStages.GROWING
        ):
            self.current_growth_stage = PlantGrowthStages.HARVESTABLE
            return True

        return False

    # Apply all automatic plant updates that occur after time advances.
    def _update_plant(self) -> bool:
        """Update starting stage, lifecycle stage, and health."""
        plant_started = self._start_plant()
        plant_stage_updated = self._update_plant_stage_by_age()
        plant_health_updated = self._update_plant_health()
        return plant_started or plant_stage_updated or plant_health_updated

    # Convert a newly planted seed into a growing plant on its first update.
    def _start_plant(self) -> bool:
        """Change a SEED into GROWING and report whether a change occurred."""
        if self.current_growth_stage == PlantGrowthStages.SEED:
            self.current_growth_stage = PlantGrowthStages.GROWING
            return True
        return False

    # Adjust plant health according to whether pH and EC are in acceptable ranges.
    def _update_plant_health(self) -> bool:
        """Apply minute-scale health damage or recovery and report any change."""
        if self.current_growth_stage == PlantGrowthStages.DEAD:
            return False

        previous_health = self.plant_health_percent
        health_changed = False

        if not self.species.ec_range[0] <= self.current_ec_level <= self.species.ec_range[1]:
            self.plant_health_percent -= 5.0 / MINUTES_PER_DAY
            health_changed = True

        if not self.species.ph_range[0] <= self.current_ph_level <= self.species.ph_range[1]:
            self.plant_health_percent -= 5.0 / MINUTES_PER_DAY
            health_changed = True

        if self.plant_health_percent <= 0:
            self.kill_plant()
        elif self.plant_health_percent < 100 and not health_changed:
            self.plant_health_percent += 2.0 / MINUTES_PER_DAY

        self.plant_health_percent = max(0.0, min(100.0, self.plant_health_percent))
        return self.plant_health_percent != previous_health

    @property
    # Report the remaining whole-day difference to harvest age.
    def days_until_harvest(self) -> float:
        """Return the nonnegative number of days until harvest age."""
        return max(0, self.species.harvest_age - self.current_age_days)

    @property
    # Report the remaining whole-day difference to maximum age.
    def days_until_expiration(self) -> float:
        """Return the nonnegative number of days until maximum age."""
        return max(0, self.species.max_age - self.current_age_days)

    @property
    # Compute the priority value used by the agent's route planner.
    def care_priority(self) -> float:
        """Return a higher value for plants needing more urgent attention."""
        if self.current_growth_stage in (
                PlantGrowthStages.HARVESTED,
                PlantGrowthStages.DEAD,
        ):
            return 30.0

        priority = max(0.0, 100.0 - self.plant_health_percent)

        if not self.species.ph_range[0] <= self.current_ph_level <= self.species.ph_range[1]:
            priority += 30.0

        if not self.species.ec_range[0] <= self.current_ec_level <= self.species.ec_range[1]:
            priority += 25.0

        if self.current_growth_stage == PlantGrowthStages.HARVESTABLE:
            priority += 40.0 + max(0.0, 20.0 - self.days_until_expiration)

        return priority

    @property
    # Translate the plant's current state into high-level requested care actions.
    def required_actions(self) -> list[PlantRequest]:
        """Return the list of care requests currently exposed by this plant."""
        if self.current_growth_stage in (
                PlantGrowthStages.DEAD,
                PlantGrowthStages.HARVESTED,
        ):
            return [PlantRequest.REPLACE]

        actions: list[PlantRequest] = []

        if self.current_growth_stage == PlantGrowthStages.HARVESTABLE:
            actions.append(PlantRequest.HARVEST)

        if not self.species.ph_range[0] <= self.current_ph_level <= self.species.ph_range[1]:
            actions.append(PlantRequest.ADJUST_PH)

        if not self.species.ec_range[0] <= self.current_ec_level <= self.species.ec_range[1]:
            actions.append(PlantRequest.REPLENISH_NUTRIENTS)

        return actions
