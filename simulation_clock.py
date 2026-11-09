"""Simulation clock that converts real delta time into exact simulated-second ticks."""

from __future__ import annotations

import math


class SimulationClock:
    """Track the current simulated day and second-of-day with real-time compression."""

    SECONDS_PER_MINUTE = 60
    MINUTES_PER_DAY = 1440
    SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_DAY

    # Initialize the clock with a configurable simulated-time compression rate.
    def __init__(self, simulated_seconds_per_real_second: float) -> None:
        """Create a clock with day-local seconds starting at day zero."""
        if (
                not math.isfinite(simulated_seconds_per_real_second)
                or simulated_seconds_per_real_second <= 0
        ):
            raise ValueError(
                "simulated_seconds_per_real_second must be finite and greater than zero."
            )

        self.simulated_seconds_per_real_second = float(
            simulated_seconds_per_real_second
        )
        self.current_day = 0
        self.second_of_day = 0
        self._fractional_simulated_seconds = 0.0

    # Convert a real-time delta into the number of whole simulated seconds now due.
    def add_real_delta(self, delta_real_seconds: float) -> int:
        """Accumulate real elapsed time and return whole simulated seconds to process."""
        if not math.isfinite(delta_real_seconds) or delta_real_seconds < 0:
            raise ValueError(
                "delta_real_seconds must be finite and cannot be negative."
            )

        simulated_delta = (
                delta_real_seconds * self.simulated_seconds_per_real_second
        )
        if not math.isfinite(simulated_delta):
            raise ValueError("The scaled simulated-time delta is not finite.")
        self._fractional_simulated_seconds += simulated_delta
        whole_seconds = int(self._fractional_simulated_seconds)
        self._fractional_simulated_seconds -= whole_seconds
        return whole_seconds

    # Advance the clock by one second and reset second-of-day at each new day.
    def advance_one_second(self) -> None:
        """Advance exactly one simulated second and roll over at 86,400 seconds."""
        self.second_of_day += 1
        if self.second_of_day >= self.SECONDS_PER_DAY:
            self.current_day += 1
            self.second_of_day = 0

    @property
    # Derive total elapsed seconds only when another component needs an absolute value.
    def elapsed_simulated_seconds(self) -> int:
        """Return elapsed seconds derived from day and second-of-day state."""
        return self.current_day * self.SECONDS_PER_DAY + self.second_of_day

    @property
    # Preserve the previous property name as a derived compatibility alias.
    def total_simulated_seconds(self) -> int:
        """Return elapsed simulated seconds without storing an ever-growing counter."""
        return self.elapsed_simulated_seconds

    @property
    # Report the unprocessed fractional part of the next simulated second.
    def fractional_simulated_second(self) -> float:
        """Return accumulated simulated time between 0.0 and less than 1.0 second."""
        return self._fractional_simulated_seconds

    @property
    # Report elapsed simulated minutes as a floating-point value.
    def total_simulated_minutes(self) -> float:
        """Return elapsed simulated time measured in minutes."""
        return self.elapsed_simulated_seconds / self.SECONDS_PER_MINUTE

    @property
    # Report elapsed simulated days as a floating-point value.
    def total_simulated_days(self) -> float:
        """Return elapsed simulated time measured in days."""
        return self.current_day + self.second_of_day / self.SECONDS_PER_DAY

    @property
    # Preserve the earlier zero-based current-day property name.
    def current_day_index(self) -> int:
        """Return the current zero-based simulated day."""
        return self.current_day

    @property
    # Report the completed minute offset within the current simulated day.
    def minute_of_day(self) -> int:
        """Return the completed simulated minute within the current day."""
        return self.second_of_day // self.SECONDS_PER_MINUTE

    @property
    # Report the second offset within the current simulated minute.
    def second_of_minute(self) -> int:
        """Return the second within the current simulated minute."""
        return self.second_of_day % self.SECONDS_PER_MINUTE

    # Format the day-local clock as HH:MM:SS for logs and visualizations.
    def time_of_day_text(self) -> str:
        """Return the current simulated time of day as HH:MM:SS."""
        hours = self.second_of_day // 3600
        minutes = (self.second_of_day % 3600) // 60
        seconds = self.second_of_day % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Change the time compression rate without resetting elapsed simulation time.
    def set_time_scale(self, simulated_seconds_per_real_second: float) -> None:
        """Update the positive simulated-seconds-per-real-second multiplier."""
        if (
                not math.isfinite(simulated_seconds_per_real_second)
                or simulated_seconds_per_real_second <= 0
        ):
            raise ValueError(
                "simulated_seconds_per_real_second must be finite and greater than zero."
            )
        self.simulated_seconds_per_real_second = float(
            simulated_seconds_per_real_second
        )
