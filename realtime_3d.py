"""Interactive Matplotlib 3-D renderer for the live farm simulation."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import matplotlib
import numpy as np
from matplotlib.backend_bases import Event
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D, Path3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from farm import Farm3D
from plant_class import PlantGrowthStages
from project_info import PROJECT_NAME
from simulation import FarmSimulation

SPECIES_COLOR_PALETTE = (
    "tab:green",
    "tab:blue",
    "tab:orange",
    "tab:purple",
    "tab:red",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "gold",
)

STAGE_MARKERS = {
    PlantGrowthStages.SEED: "s",
    PlantGrowthStages.GROWING: "o",
    PlantGrowthStages.HARVESTABLE: "^",
    PlantGrowthStages.HARVESTED: "x",
    PlantGrowthStages.DEAD: "X",
}


def _select_interactive_backend() -> str:
    """Select a desktop GUI backend before importing matplotlib.pyplot."""
    requested = os.environ.get("MPLBACKEND", "").strip()

    if requested:
        lowered = requested.casefold()
        if lowered in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
            raise RuntimeError(
                "MPLBACKEND is set to a non-interactive backend "
                f"({requested}). Remove that environment variable or set it "
                "to TkAgg before starting the 3-D renderer."
            )

        matplotlib.use(requested, force=True)
        return requested

    # TkAgg is the most reliable option for a normal Windows Python install.
    try:
        __import__("tkinter")

        matplotlib.use("TkAgg", force=True)
        return "TkAgg"
    except (ImportError, RuntimeError) as tk_error:
        # Fall back to Qt when available.
        for backend, module_name in (
                ("QtAgg", "PyQt6"),
                ("QtAgg", "PySide6"),
                ("Qt5Agg", "PyQt5"),
        ):
            try:
                __import__(module_name)
                matplotlib.use(backend, force=True)
                return backend
            except (ImportError, RuntimeError):
                continue

        raise RuntimeError(
            "No interactive Matplotlib GUI backend is available. "
            "Install Tk support for Python or install PyQt6, then rerun. "
            f"Tk backend error: {tk_error}"
        ) from tk_error


ACTIVE_MATPLOTLIB_BACKEND = _select_interactive_backend()
from matplotlib import pyplot as plt  # noqa: E402


@dataclass
class Realtime3DSettings:
    """Configuration for interactive real-time 3-D observation."""

    simulation_speed: float = 10.0
    frames_per_second: int = 30
    floor_spacing: float = 2.5
    plant_marker_size: float = 45.0
    agent_marker_size: float = 180.0


class Realtime3DRenderer:
    """Advance one actual FarmSimulation and render its state interactively in 3-D."""

    # Initialize the renderer around the exact simulation instance that will be displayed.
    def __init__(
            self,
            simulation: FarmSimulation,
            settings: Optional[Realtime3DSettings] = None,
    ) -> None:
        """Create the interactive figure and initialize playback state."""
        self.simulation = simulation
        self.settings = settings or Realtime3DSettings()
        if (
                not math.isfinite(self.settings.simulation_speed)
                or self.settings.simulation_speed <= 0
        ):
            raise ValueError("simulation_speed must be finite and greater than zero.")
        if (
                not math.isfinite(self.settings.frames_per_second)
                or self.settings.frames_per_second <= 0
        ):
            raise ValueError(
                "frames_per_second must be finite and greater than zero."
            )
        for setting_name, setting_value in (
                ("floor_spacing", self.settings.floor_spacing),
                ("plant_marker_size", self.settings.plant_marker_size),
                ("agent_marker_size", self.settings.agent_marker_size),
        ):
            if not math.isfinite(setting_value) or setting_value <= 0:
                raise ValueError(
                    f"{setting_name} must be finite and greater than zero."
                )

        self.paused = False
        self._previous_real_time = time.perf_counter()
        self._figure = plt.figure(figsize=(14, 9))
        self._axis = cast(
            Axes3D,
            self._figure.add_subplot(111, projection="3d"),
        )
        self._figure.subplots_adjust(
            left=0.04,
            right=0.67,
            bottom=0.20,
            top=0.93,
        )
        self._species_colors = self.species_color_map(simulation.farm)
        current_species_counts = simulation.farm.species_counts()
        self._initial_species_counts = {
            species_name: current_species_counts.get(species_name, 0)
            for species_name in self._species_colors
        }
        self._initial_dead_species_by_position = {
            position: simulation.farm.get_plant(position).species.name
            for position in simulation.farm.positions()
            if simulation.farm.get_plant(position).current_growth_stage
               == PlantGrowthStages.DEAD
        }
        self._plant_artists: dict[
            tuple[str, PlantGrowthStages], Path3DCollection
        ] = {}
        self._initialize_artists()
        self._figure.canvas.mpl_connect("key_press_event", self.handle_key_press)

    # Create all plot artists once so subsequent frames only replace their data.
    def _initialize_artists(self) -> None:
        """Create persistent grouped plant, route, target, and agent artists."""
        ax = self._axis
        farm = self.simulation.farm

        for species_name, color in self._species_colors.items():
            for stage, marker in STAGE_MARKERS.items():
                artist = ax.scatter(
                    [], [], [], marker=marker, color=color, s=[]
                )
                self._plant_artists[(species_name, stage)] = cast(
                    Path3DCollection,
                    artist,
                )

        self._route_artist = cast(
            Line3D,
            ax.plot(
                [], [], [], linestyle="--", linewidth=1.5, color="black"
            )[0],
        )
        self._target_artist = cast(
            Path3DCollection,
            ax.scatter(
                [], [], [], marker="*", s=[], facecolors="none",
                edgecolors="black", linewidths=2,
            ),
        )
        self._agent_artist = cast(
            Path3DCollection,
            ax.scatter(
                [], [], [], marker="P", s=[], edgecolors="black",
                color="black", linewidths=1.3,
            ),
        )

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_zlabel("Floor")
        ax.set_xlim(-0.5, max(0.5, farm.columns - 0.5))
        ax.set_ylim(-0.5, max(0.5, farm.rows - 0.5))
        max_z = max(
            self.settings.floor_spacing,
            (farm.floors - 1) * self.settings.floor_spacing,
        )
        ax.set_zlim(-0.5, max_z + 0.5)
        if farm.floors > 1:
            ax.set_zticks(
                [
                    floor * self.settings.floor_spacing
                    for floor in range(farm.floors)
                ],
                [str(floor) for floor in range(farm.floors)],
            )

        self._species_legend = self._add_legends(
            ax,
            self._species_colors,
            STAGE_MARKERS,
        )
        self._status_artist = self._figure.text(
            0.04,
            0.13,
            "",
            verticalalignment="top",
            horizontalalignment="left",
        )
        self._figure.text(
            0.98,
            0.13,
            "Controls\n"
            "Space = pause/resume | Esc = close\n"
            "1 = real-time | 2 = 1 min/sec | 3 = 1 hr/sec | 4 = 1 day/sec",
            verticalalignment="top",
            horizontalalignment="right",
        )
        self._update_harvest_efficiency()

    @staticmethod
    def _update_scatter(
            artist: Path3DCollection,
            x_values: list[float],
            y_values: list[float],
            z_values: list[float],
            sizes: list[float],
    ) -> None:
        """Replace one persistent 3-D scatter collection's point data."""
        offsets = (
            np.column_stack((x_values, y_values))
            if x_values
            else np.empty((0, 2), dtype=float)
        )
        artist.set_offsets(offsets)
        # Matplotlib accepts array-like z values, but its 3-D type metadata
        # incorrectly rejects both list[float] and ndarray in some IDEs.
        z_data = cast(Any, np.asarray(z_values, dtype=float))
        artist.set_3d_properties(z_data, "z")
        # Path3DCollection indexes this internal array with NumPy indices
        # during projection, so it must not be left as a Python list.
        artist.set_sizes(np.asarray(sizes, dtype=float))

    @property
    def plant_collection_count(self) -> int:
        """Return the fixed number of species-and-stage plant collections."""
        return len(self._plant_artists)

    @property
    def harvest_efficiency_text(self) -> str:
        """Return the currently displayed per-species harvest summary."""
        lines = [self._species_legend.get_title().get_text()]
        lines.extend(
            text.get_text() for text in self._species_legend.get_texts()
        )
        return "\n".join(lines)

    # Update the persistent per-species efficiency panel from completed actions.
    def _update_harvest_efficiency(self) -> None:
        """Show harvested, dead, and outcome efficiency for every species."""
        harvested_counts = {
            species_name: 0 for species_name in self._initial_species_counts
        }
        dead_counts = {
            species_name: 0 for species_name in self._initial_species_counts
        }
        for species_name in self._initial_dead_species_by_position.values():
            dead_counts[species_name] = dead_counts.get(species_name, 0) + 1

        unmatched_initial_dead = set(self._initial_dead_species_by_position)
        for record in self.simulation.agent.action_records:
            species_name = record["species"]
            if record["action"] == "HARVEST":
                harvested_counts[species_name] = (
                        harvested_counts.get(species_name, 0) + 1
                )
            elif record["action"] == "REMOVE_DEAD":
                position = record["position"]
                if position in unmatched_initial_dead:
                    unmatched_initial_dead.remove(position)
                else:
                    dead_counts[species_name] = dead_counts.get(species_name, 0) + 1

        legend_texts = self._species_legend.get_texts()
        for text, species_name in zip(
                legend_texts,
                self._initial_species_counts,
        ):
            harvested = harvested_counts.get(species_name, 0)
            dead = dead_counts.get(species_name, 0)
            completed_outcomes = harvested + dead
            percentage = (
                harvested / completed_outcomes * 100.0
                if completed_outcomes > 0
                else 0.0
            )
            text.set_text(
                f"{species_name}: Harvested {harvested} | Dead {dead} | "
                f"Efficiency {percentage:.2f}%"
            )

    # Assign each selected species a stable color in farm selection order.
    @staticmethod
    def species_color_map(farm: Farm3D) -> dict[str, str]:
        """Return a stable display color for every species in the farm."""
        species_names = list(
            dict.fromkeys(species.name for species in farm.species_options)
        )
        return {
            name: SPECIES_COLOR_PALETTE[index % len(SPECIES_COLOR_PALETTE)]
            for index, name in enumerate(species_names)
        }

    # Add separate legends for species colors and marker meanings.
    @staticmethod
    def _add_legends(
            ax: Axes3D,
            species_colors: dict[str, str],
            stage_markers: dict[PlantGrowthStages, str],
    ) -> Legend:
        """Add right-side legends and return the combined species legend."""
        species_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=8,
                label=species_name,
            )
            for species_name, color in species_colors.items()
        ]
        species_legend = ax.legend(
            handles=species_handles,
            title=(
                "Plant Species / Harvest Efficiency\n"
                "Efficiency = Harvested / (Harvested + Dead)"
            ),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.add_artist(species_legend)

        shape_handles = [
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="None",
                color="black",
                markersize=8,
                label=stage.name.replace("_", " ").title(),
            )
            for stage, marker in stage_markers.items()
        ]
        shape_handles.extend(
            (
                Line2D(
                    [],
                    [],
                    marker="P",
                    linestyle="None",
                    color="black",
                    markersize=9,
                    label="Agent",
                ),
                Line2D(
                    [],
                    [],
                    marker="*",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=11,
                    label="Current target",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=11,
                    label="Larger marker: care request",
                ),
                Line2D(
                    [],
                    [],
                    linestyle="--",
                    color="black",
                    label="Actual movement path",
                ),
            )
        )
        ax.legend(
            handles=shape_handles,
            title="Marker shapes",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
        )
        return species_legend

    # Convert a farm coordinate into renderer x/y/z space.
    def _to_xyz(
            self,
            farm_position: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Map (floor, row, column) into (x, y, z) plot coordinates."""
        floor, row, column = farm_position
        return (
            column,
            row,
            floor * self.settings.floor_spacing,
        )

    # Advance the simulation from measured real delta time and redraw one frame.
    def update_frame(self) -> None:
        """Advance the same simulation instance from wall-clock delta time and redraw."""
        current_real_time = time.perf_counter()
        delta_real_seconds = current_real_time - self._previous_real_time
        self._previous_real_time = current_real_time

        if not self.paused:
            simulated_seconds = (
                    delta_real_seconds * self.settings.simulation_speed
            )
            self.simulation.update(simulated_seconds)

        self.draw()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    # Draw the farm, plant states, agent, target, and lookahead route.
    def draw(self) -> None:
        """Render the current simulation state without creating a second simulation."""
        ax = self._axis
        farm = self.simulation.farm
        agent = self.simulation.agent
        task = agent.current_task
        request_positions = {
            request.position for request in farm.collect_requests()
        }
        grouped_points: dict[
            tuple[str, PlantGrowthStages],
            tuple[list[float], list[float], list[float], list[float]],
        ] = {
            key: ([], [], [], []) for key in self._plant_artists
        }

        # Group locations so each species-and-stage combination uses one collection.
        for position in farm.positions():
            plant = farm.get_plant(position)
            x, y, z = self._to_xyz(position)
            size = self.settings.plant_marker_size
            if position in request_positions:
                size *= 1.8

            x_values, y_values, z_values, sizes = grouped_points[
                (plant.species.name, plant.current_growth_stage)
            ]
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
            sizes.append(size)

        for key, artist in self._plant_artists.items():
            self._update_scatter(
                artist,
                *grouped_points[key],
            )

        # Draw the exact axis-aligned cells used by movement and lookahead travel.
        interpolated_agent_position = agent.render_position_at(
            self.simulation.clock.fractional_simulated_second
        )
        agent_xyz = self._to_xyz(interpolated_agent_position)
        route_positions = [agent_xyz]
        route_positions.extend(
            self._to_xyz(position)
            for position in agent.planned_route_cells()
        )

        if len(route_positions) > 1:
            self._route_artist.set_data_3d(
                [point[0] for point in route_positions],
                [point[1] for point in route_positions],
                [point[2] for point in route_positions],
            )
        else:
            self._route_artist.set_data_3d([], [], [])

        # Draw the current committed task separately.
        if task is not None:
            target_x, target_y, target_z = self._to_xyz(task.position)
            self._update_scatter(
                self._target_artist,
                [target_x],
                [target_y],
                [target_z],
                [280.0],
            )
        else:
            self._update_scatter(self._target_artist, [], [], [], [])

        # Draw the continuously interpolated agent position.
        agent_x, agent_y, agent_z = agent_xyz
        self._update_scatter(
            self._agent_artist,
            [agent_x],
            [agent_y],
            [agent_z],
            [self.settings.agent_marker_size],
        )
        self._update_harvest_efficiency()

        current_day = self.simulation.clock.current_day
        second_of_day = self.simulation.clock.second_of_day
        hours = second_of_day // 3600
        minutes = (second_of_day % 3600) // 60
        seconds = second_of_day % 60

        current_species = "-"
        if task is not None:
            current_species = farm.get_plant(
                task.position
            ).species.name

        ax.set_title(f"{PROJECT_NAME} — Real-Time 3-D Simulation")
        self._status_artist.set_text(
            f"Day {current_day}  {hours:02d}:{minutes:02d}:{seconds:02d}  |  "
            f"Speed {self.settings.simulation_speed:g}x  |  "
            f"Action {'PAUSED' if self.paused else agent.current_action_name}\n"
            "Agent position "
            f"{tuple(round(v, 2) for v in interpolated_agent_position)}  |  "
            f"Species {current_species}  |  "
            f"Tasks {agent.completed_tasks}  |  "
            f"Backlog {len(request_positions)}"
        )

    # Handle pause and simulation-speed controls from the keyboard.
    def handle_key_press(self, event: Event) -> None:
        """Change pause or playback speed in response to keyboard input."""
        key = getattr(event, "key", None)
        if key == "escape":
            self.close()
        elif key == " ":
            self.paused = not self.paused
        elif key == "1":
            self.settings.simulation_speed = 1.0
        elif key == "2":
            self.settings.simulation_speed = 60.0
        elif key == "3":
            self.settings.simulation_speed = 3600.0
        elif key == "4":
            self.settings.simulation_speed = 86400.0

    # Close this renderer's Matplotlib figure.
    def close(self) -> None:
        """Close the interactive 3-D window."""
        plt.close(self._figure)

    # Start the interactive renderer and keep running until the visualization window closes.
    def run(self, max_frames: Optional[int] = None) -> None:
        """Open the 3-D window and continuously run until it is closed by the user."""
        backend = matplotlib.get_backend()
        if backend.casefold() in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
            raise RuntimeError(
                "The 3-D renderer is using a non-interactive Matplotlib backend "
                f"({backend}), so no window can be displayed."
            )

        frame_interval = 1.0 / self.settings.frames_per_second
        rendered_frames = 0

        self.draw()
        manager = self._figure.canvas.manager
        if manager is None:
            raise RuntimeError("The 3-D figure does not have a window manager.")
        manager.set_window_title(
            f"{PROJECT_NAME} — Real-Time 3-D Simulation"
        )
        figure_number = manager.num
        plt.show(block=False)
        self._previous_real_time = time.perf_counter()

        while plt.fignum_exists(figure_number):
            frame_start = time.perf_counter()
            self.update_frame()
            rendered_frames += 1

            if max_frames is not None and rendered_frames >= max_frames:
                break

            frame_elapsed = time.perf_counter() - frame_start
            remaining = frame_interval - frame_elapsed
            plt.pause(max(0.001, remaining))

        if max_frames is not None and plt.fignum_exists(figure_number):
            plt.close(self._figure)

    # Render the current state to an image file for testing or documentation.
    def save_snapshot(self, output_path: str) -> None:
        """Draw the current simulation state and save one static 3-D image."""
        self.draw()
        self._figure.savefig(output_path, dpi=140, bbox_inches="tight")
