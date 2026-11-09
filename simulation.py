"""Authoritative simulation runtime that advances the farm and agent together."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Union

from agent import FarmAgent
from farm import Farm3D
from simulation_clock import SimulationClock

SimulationObserver = Callable[["FarmSimulation"], None]


class FarmSimulation:
    """Own the single authoritative run used by both the program and visualizer."""

    # Initialize one simulation around an already-created farm, agent, and timescale.
    def __init__(
            self,
            farm: Farm3D,
            agent: FarmAgent,
            simulated_seconds_per_real_second: float = 1440.0,
    ) -> None:
        """Create a simulation and verify that the agent belongs to this farm."""
        if agent.farm is not farm:
            raise ValueError("Agent must belong to the supplied farm.")

        self.farm = farm
        self.agent = agent
        self.clock = SimulationClock(simulated_seconds_per_real_second)
        self._observers: list[SimulationObserver] = []

    @property
    # Preserve a convenient compatibility alias for code that needs total simulated seconds.
    def simulated_seconds(self) -> int:
        """Return total elapsed simulated seconds from the authoritative clock."""
        return self.clock.total_simulated_seconds

    # Register a callback that receives this exact simulation after each simulated second.
    def add_observer(self, observer: SimulationObserver) -> None:
        """Attach an observer that sees the actual run state after every simulated second."""
        self._observers.append(observer)

    # Remove a previously registered simulation observer.
    def remove_observer(self, observer: SimulationObserver) -> None:
        """Detach an observer when it is no longer needed."""
        self._observers.remove(observer)

    # Convert one real-time delta into however many exact simulated-second ticks are due.
    def update(self, delta_real_seconds: float) -> int:
        """Advance simulation time according to real delta time and return ticks processed."""
        seconds_to_process = self.clock.add_real_delta(delta_real_seconds)

        for _ in range(seconds_to_process):
            self.step_one_simulated_second()

        return seconds_to_process

    # Advance the authoritative simulation by exactly one simulated second.
    def step_one_simulated_second(self) -> None:
        """Advance time one second, process minute boundaries, update the agent, and notify observers."""
        self.clock.advance_one_second()
        current_second = self.clock.total_simulated_seconds

        if current_second % SimulationClock.SECONDS_PER_MINUTE == 0:
            events = self.farm.simulate_minute()
            self.agent.on_farm_events(events, current_second)

        if current_second % SimulationClock.SECONDS_PER_DAY == 0:
            self.agent.on_safety_check(current_second)

        self.agent.update_one_second(current_second)
        self._notify_observers()

    # Preserve the prior step name as an alias for one exact simulated-second tick.
    def step(self) -> None:
        """Advance the authoritative simulation by exactly one simulated second."""
        self.step_one_simulated_second()

    # Run a fixed number of simulated seconds without waiting for wall-clock time.
    def run_simulated_seconds(self, seconds: int) -> None:
        """Advance the simulation deterministically for a nonnegative number of simulated seconds."""
        if seconds < 0:
            raise ValueError("Simulation duration cannot be negative.")

        for _ in range(seconds):
            self.step_one_simulated_second()

    # Preserve the prior run method for deterministic tests and batch processing.
    def run(self, seconds: int) -> None:
        """Advance the simulation for a fixed number of simulated seconds."""
        self.run_simulated_seconds(seconds)

    # Run until the requested simulated-day target using measured real delta time.
    def run_realtime(
            self,
            total_simulated_days: float,
            sleep_seconds: float = 0.001,
    ) -> None:
        """Run with perf_counter delta time until the requested simulated duration is reached."""
        if not math.isfinite(total_simulated_days) or total_simulated_days <= 0:
            raise ValueError(
                "total_simulated_days must be finite and greater than zero."
            )
        if not math.isfinite(sleep_seconds) or sleep_seconds < 0:
            raise ValueError("sleep_seconds must be finite and cannot be negative.")

        target_seconds = int(
            round(total_simulated_days * SimulationClock.SECONDS_PER_DAY)
        )
        previous_real_time = time.perf_counter()

        while self.clock.total_simulated_seconds < target_seconds:
            current_real_time = time.perf_counter()
            delta_real_seconds = current_real_time - previous_real_time
            previous_real_time = current_real_time

            remaining_seconds = (
                    target_seconds - self.clock.total_simulated_seconds
            )
            seconds_due = self.clock.add_real_delta(delta_real_seconds)
            seconds_due = min(seconds_due, remaining_seconds)

            for _ in range(seconds_due):
                self.step_one_simulated_second()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    # Send the current state to every observer attached to this exact run.
    def _notify_observers(self) -> None:
        """Call each observer with this simulation instance."""
        for observer in tuple(self._observers):
            observer(self)

    # Return summary statistics for the completed or partially completed run.
    def summary(
            self,
    ) -> dict[str, Union[int, float, tuple[int, int, int]]]:
        """Return key time and agent statistics for display or testing."""
        return {
            "simulated_seconds": self.clock.total_simulated_seconds,
            "simulated_minutes": self.clock.total_simulated_minutes,
            "simulated_days": self.clock.total_simulated_days,
            "agent_position": self.agent.position,
            "distance_traveled": self.agent.distance_traveled,
            "completed_tasks": self.agent.completed_tasks,
            "completed_actions": self.agent.completed_actions,
        }
