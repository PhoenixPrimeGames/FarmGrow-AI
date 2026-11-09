"""GIF recording layer that observes one actual FarmSimulation run."""

from __future__ import annotations

import math
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union, cast

from PIL import Image, ImageDraw

from farm import Farm3D, Position
from project_info import PROJECT_NAME
from simulation import FarmSimulation
from simulation_clock import SimulationClock

SECONDS_PER_DAY = SimulationClock.SECONDS_PER_DAY
MAX_DAYS_PER_GIF = 30
CELL = 58
MARGIN = 30
HEADER = 140
FLOOR_GAP = 35
FOOTER = 40
GIF_MAX_DIMENSION = 65_535
DEFAULT_DETAIL_MEMORY_LIMIT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class PlantFrameState:
    """Immutable per-plant state copied from the actual run for one GIF frame."""

    species_name: str
    growth_stage: str
    has_request: bool


@dataclass(frozen=True)
class FrameSnapshot:
    """Immutable visualization snapshot captured from the authoritative simulation."""

    elapsed_second: int
    display_day: int
    time_of_day: str
    agent_position: Position
    current_task_position: Optional[Position]
    current_task_species: Optional[str]
    current_action: str
    lookahead_plan: tuple[Position, ...]
    completed_tasks: int
    completed_actions: int
    distance_traveled: int
    action_totals: dict[str, int]
    last_action_text: str
    plants: dict[Position, PlantFrameState]


class GifVisualizerRecorder:
    """Observe one simulation and create overview plus slower detailed GIF clips."""

    # Configure overview chunks and selected detailed activity days.
    def __init__(
            self,
            total_run_days: int,
            output_directory: Union[str, Path],
            max_frames_per_gif: int = 180,
            frame_duration_ms: int = 240,
            detail_days: Optional[tuple[int, ...]] = None,
            detail_interval_seconds: int = 600,
            detail_frame_duration_ms: int = 500,
            detail_memory_limit_bytes: int = DEFAULT_DETAIL_MEMORY_LIMIT_BYTES,
    ) -> None:
        """Initialize actual-run recording settings for overview and detailed GIFs."""
        if total_run_days <= 0:
            raise ValueError("Total run days must be greater than zero.")
        if max_frames_per_gif <= 1:
            raise ValueError("max_frames_per_gif must be greater than one.")
        if detail_interval_seconds <= 0:
            raise ValueError("detail_interval_seconds must be greater than zero.")
        if frame_duration_ms <= 0 or detail_frame_duration_ms <= 0:
            raise ValueError("Frame durations must be greater than zero.")
        if detail_memory_limit_bytes <= 0:
            raise ValueError("detail_memory_limit_bytes must be greater than zero.")

        self.total_run_days = total_run_days
        self.total_run_seconds = total_run_days * SECONDS_PER_DAY
        self.output_directory = Path(output_directory)
        self.max_frames_per_gif = max_frames_per_gif
        self.frame_duration_ms = frame_duration_ms
        self.chunk_seconds = MAX_DAYS_PER_GIF * SECONDS_PER_DAY
        self.detail_interval_seconds = detail_interval_seconds
        self.detail_frame_duration_ms = detail_frame_duration_ms
        self.detail_memory_limit_bytes = detail_memory_limit_bytes
        self.detail_days = (
            detail_days
            if detail_days is not None
            else self._default_detail_days(total_run_days)
        )
        self._overview_frames: dict[int, list[FrameSnapshot]] = {}
        self._detail_frames: dict[int, list[FrameSnapshot]] = {
            day: [] for day in self.detail_days
        }
        self._detail_last_action_count: dict[int, int] = {
            day: 0 for day in self.detail_days
        }
        self._detail_memory_bytes = 0
        self._detail_frame_order: deque[tuple[int, int, int]] = deque()
        self._validated_farm_identity: Optional[int] = None

    # Choose representative one-based days across the run for slower inspection clips.
    @staticmethod
    def _default_detail_days(total_run_days: int) -> tuple[int, ...]:
        """Return unique representative days including the start and end of the run."""
        candidates = (1, 30, 60, 90, total_run_days)
        return tuple(
            day for day in dict.fromkeys(candidates)
            if 1 <= day <= total_run_days
        )

    # Capture the initial t=0 state from the same simulation that will be run.
    def capture_initial_state(self, simulation: FarmSimulation) -> None:
        """Record the actual simulation state before the first second advances."""
        self.validate_farm(simulation.farm)
        snapshot = self._snapshot(simulation)
        self._overview_frames.setdefault(0, []).append(snapshot)
        if 1 in self._detail_frames:
            self._append_detail_snapshot(1, snapshot)

    # Calculate the exact canvas dimensions produced by the floor-slice renderer.
    @staticmethod
    def frame_dimensions(farm: Farm3D) -> tuple[int, int]:
        """Return the GIF frame width and height for one farm layout."""
        floor_width = farm.columns * CELL
        floor_height = farm.rows * CELL
        width = (
                MARGIN * 2
                + farm.floors * floor_width
                + (farm.floors - 1) * FLOOR_GAP
        )
        height = HEADER + floor_height + MARGIN + FOOTER
        return width, height

    # Reject layouts that cannot be represented by the GIF file format.
    def validate_farm(self, farm: Farm3D) -> None:
        """Raise ValueError when the rendered farm exceeds GIF dimensions."""
        width, height = self.frame_dimensions(farm)
        if width > GIF_MAX_DIMENSION or height > GIF_MAX_DIMENSION:
            raise ValueError(
                "Farm visualization would be "
                f"{width:,}x{height:,} pixels, but GIF dimensions cannot exceed "
                f"{GIF_MAX_DIMENSION:,} pixels. Reduce floors, rows, or columns."
            )
        self._validated_farm_identity = id(farm)

    # Validate once for observers that were attached without an initial capture.
    def _ensure_farm_validated(self, farm: Farm3D) -> None:
        """Validate a farm only when this recorder sees a new instance."""
        if self._validated_farm_identity != id(farm):
            self.validate_farm(farm)

    # Report the number of captured detail frames for diagnostics and tests.
    def detail_frame_count(self, day: int) -> int:
        """Return the number of detail frames currently stored for one day."""
        frames = self._detail_frames.get(day)
        return len(frames) if frames is not None else 0

    @property
    def detail_memory_bytes(self) -> int:
        """Return the estimated deep memory retained by detail snapshots."""
        return self._detail_memory_bytes

    # Receive the same FarmSimulation instance after each authoritative simulation step.
    def observe(self, simulation: FarmSimulation) -> None:
        """Sample overview and detailed frames directly from the actual run."""
        self._ensure_farm_validated(simulation.farm)
        elapsed = simulation.clock.elapsed_simulated_seconds
        chunk_index = self._chunk_index_for_second(elapsed)
        overview_interval = self._sample_interval_for_chunk(chunk_index)
        overview_frames = self._overview_frames.setdefault(chunk_index, [])

        force_final = elapsed == self.total_run_seconds
        if force_final or not overview_frames or elapsed % overview_interval == 0:
            self._append_unique(overview_frames, self._snapshot(simulation))

        if (
                simulation.clock.current_day > 0
                and simulation.clock.second_of_day == 0
        ):
            one_based_day = simulation.clock.current_day
        else:
            one_based_day = simulation.clock.current_day + 1

        if one_based_day in self._detail_frames:
            action_count = simulation.agent.completed_actions
            action_changed = (
                    action_count != self._detail_last_action_count[one_based_day]
            )
            harvest_or_plant_completed = False
            if action_changed and simulation.agent.action_records:
                harvest_or_plant_completed = (
                        str(simulation.agent.action_records[-1]["action"])
                        in {"HARVEST", "PLANT"}
                )
            if (
                    force_final
                    or harvest_or_plant_completed
                    or simulation.clock.second_of_day % self.detail_interval_seconds == 0
            ):
                self._append_detail_snapshot(
                    one_based_day,
                    self._snapshot(simulation),
                )
                self._detail_last_action_count[one_based_day] = action_count

    # Append one detail frame and evict oldest frames until under the memory cap.
    def _append_detail_snapshot(
            self,
            day: int,
            snapshot: FrameSnapshot,
    ) -> None:
        """Retain a unique detail snapshot within the rolling memory budget."""
        frames = self._detail_frames[day]
        if frames and frames[-1].elapsed_second == snapshot.elapsed_second:
            return

        snapshot_bytes = self._deep_size(snapshot)
        if snapshot_bytes > self.detail_memory_limit_bytes:
            return

        frames.append(snapshot)
        self._detail_memory_bytes += snapshot_bytes
        self._detail_frame_order.append(
            (day, snapshot.elapsed_second, snapshot_bytes)
        )

        while (
                self._detail_memory_bytes > self.detail_memory_limit_bytes
                and self._detail_frame_order
        ):
            old_day, old_elapsed, old_bytes = self._detail_frame_order.popleft()
            old_frames = self._detail_frames[old_day]
            for index, old_snapshot in enumerate(old_frames):
                if old_snapshot.elapsed_second == old_elapsed:
                    old_frames.pop(index)
                    self._detail_memory_bytes -= old_bytes
                    break

    # Estimate complete retained memory for nested snapshot containers.
    @staticmethod
    def _deep_size(value: object, seen: Optional[set[int]] = None) -> int:
        """Return a conservative recursive size estimate without double counting."""
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)

        total = sys.getsizeof(value)
        if isinstance(value, Mapping):
            for key, item in value.items():
                total += GifVisualizerRecorder._deep_size(key, seen)
                total += GifVisualizerRecorder._deep_size(item, seen)
        elif isinstance(value, (list, tuple, set, frozenset, deque)):
            for item in value:
                total += GifVisualizerRecorder._deep_size(item, seen)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(cast(Any, value)):
                total += GifVisualizerRecorder._deep_size(
                    getattr(value, field.name),
                    seen,
                )
        return total

    # Append a snapshot unless that exact simulation second is already the final frame.
    @staticmethod
    def _append_unique(
            frames: list[FrameSnapshot],
            snapshot: FrameSnapshot,
    ) -> None:
        """Prevent duplicate frames at shared sampling and boundary points."""
        if not frames or frames[-1].elapsed_second != snapshot.elapsed_second:
            frames.append(snapshot)

    # Convert the mutable current simulation state into a complete immutable frame.
    @staticmethod
    def _snapshot(simulation: FarmSimulation) -> FrameSnapshot:
        """Copy all farm, agent, and action-summary data needed to render a frame."""
        request_positions = {
            request.position for request in simulation.farm.collect_requests()
        }
        plants = {
            position: PlantFrameState(
                species_name=simulation.farm.get_plant(position).species.name,
                growth_stage=simulation.farm.get_plant(position).current_growth_stage.name,
                has_request=position in request_positions,
            )
            for position in simulation.farm.positions()
        }

        current_task = simulation.agent.current_task
        current_task_position = (
            current_task.position if current_task is not None else None
        )
        current_task_species = (
            simulation.farm.get_plant(current_task_position).species.name
            if current_task_position is not None
            else None
        )

        action_totals: dict[str, int] = {}
        for record in simulation.agent.action_records:
            action = str(record["action"])
            action_totals[action] = action_totals.get(action, 0) + 1

        if simulation.agent.action_records:
            last = simulation.agent.action_records[-1]
            last_action_text = (
                f"{last['action']} {last['species']} at {last['position']}"
            )
        else:
            last_action_text = "None"

        if simulation.clock.second_of_day == 0 and simulation.clock.current_day > 0:
            display_day = simulation.clock.current_day
            time_of_day = "24:00:00"
        else:
            display_day = simulation.clock.current_day + 1
            time_of_day = simulation.clock.time_of_day_text()

        return FrameSnapshot(
            elapsed_second=simulation.clock.elapsed_simulated_seconds,
            display_day=display_day,
            time_of_day=time_of_day,
            agent_position=simulation.agent.position,
            current_task_position=current_task_position,
            current_task_species=current_task_species,
            current_action=simulation.agent.current_action_name,
            lookahead_plan=tuple(simulation.agent.lookahead_plan),
            completed_tasks=simulation.agent.completed_tasks,
            completed_actions=simulation.agent.completed_actions,
            distance_traveled=simulation.agent.distance_traveled,
            action_totals=action_totals,
            last_action_text=last_action_text,
            plants=plants,
        )

    # Map an elapsed second to a zero-based 30-day overview GIF chunk.
    def _chunk_index_for_second(self, elapsed_second: int) -> int:
        """Return the 30-day output chunk index for an elapsed simulation second."""
        if elapsed_second <= 0:
            return 0
        return (elapsed_second - 1) // self.chunk_seconds

    # Compute a sampling interval that keeps an overview GIF within its frame limit.
    def _sample_interval_for_chunk(self, chunk_index: int) -> int:
        """Return simulated seconds between overview frames for one chunk."""
        chunk_start = chunk_index * self.chunk_seconds
        chunk_end = min(
            (chunk_index + 1) * self.chunk_seconds,
            self.total_run_seconds,
        )
        chunk_duration = max(1, chunk_end - chunk_start)
        return max(1, math.ceil(chunk_duration / self.max_frames_per_gif))

    # Render all captured overview and detailed clips after the actual run completes.
    def write_gifs(self, simulation: FarmSimulation) -> list[Path]:
        """Write overview and slower detail GIFs and return every generated path."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []

        for chunk_index in sorted(self._overview_frames):
            frames = self._overview_frames[chunk_index]
            if not frames:
                continue
            start_day = chunk_index * MAX_DAYS_PER_GIF + 1
            end_day = min(
                (chunk_index + 1) * MAX_DAYS_PER_GIF,
                self.total_run_days,
            )
            path = self.output_directory / (
                f"overview_days_{start_day:03d}-{end_day:03d}.gif"
            )
            self._save_frames(simulation, frames, path, self.frame_duration_ms)
            output_paths.append(path)

        for day in self.detail_days:
            frames = self._detail_frames.get(day, [])
            if not frames:
                continue
            path = self.output_directory / f"detail_day_{day:03d}.gif"
            self._save_frames(
                simulation,
                frames,
                path,
                self.detail_frame_duration_ms,
            )
            output_paths.append(path)

        return output_paths

    # Render and save one list of actual-run snapshots as an animated GIF.
    def _save_frames(
            self,
            simulation: FarmSimulation,
            frames: list[FrameSnapshot],
            output_path: Path,
            duration_ms: int,
    ) -> None:
        """Render snapshots and save them as one looping GIF."""
        images = [self._render_frame(simulation, snapshot) for snapshot in frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

    # Draw one sampled state as side-by-side floor slices of the 3-D farm.
    @staticmethod
    def _render_frame(
            simulation: FarmSimulation,
            snapshot: FrameSnapshot,
    ) -> Image.Image:
        """Render one immutable actual-run snapshot to a PIL image."""
        farm = simulation.farm
        width, height = GifVisualizerRecorder.frame_dimensions(farm)
        if width > GIF_MAX_DIMENSION or height > GIF_MAX_DIMENSION:
            raise ValueError(
                "Farm visualization would be "
                f"{width:,}x{height:,} pixels, but GIF dimensions cannot exceed "
                f"{GIF_MAX_DIMENSION:,} pixels. Reduce floors, rows, or columns."
            )
        floor_width = farm.columns * CELL

        image = Image.new("1", (width, height), 1)
        draw = ImageDraw.Draw(image)

        current_plant = snapshot.current_task_species or "None"
        target_position = snapshot.current_task_position
        target_position_text = (
            "None" if target_position is None else str(target_position)
        )
        harvests = snapshot.action_totals.get("HARVEST", 0)
        plantings = snapshot.action_totals.get("PLANT", 0)
        ph_actions = snapshot.action_totals.get("ADJUST_PH", 0)
        nutrient_actions = snapshot.action_totals.get("REPLENISH_NUTRIENTS", 0)
        dead_removals = snapshot.action_totals.get("REMOVE_DEAD", 0)

        draw.text((MARGIN, 12), PROJECT_NAME, fill="black")
        draw.text(
            (MARGIN, 34),
            (
                f"Day {snapshot.display_day} | {snapshot.time_of_day} | "
                f"Action: {snapshot.current_action} | Plant: {current_plant}"
            ),
            fill="black",
        )
        draw.text(
            (MARGIN, 56),
            (
                f"Position: {snapshot.agent_position} | Target: "
                f"{target_position_text} | Lookahead: "
                f"{list(snapshot.lookahead_plan)}"
            ),
            fill="black",
        )
        draw.text(
            (MARGIN, 78),
            (
                f"Harvested: {harvests} | Planted: {plantings} | "
                f"pH: {ph_actions} | Nutrients: {nutrient_actions} | "
                f"Dead removed: {dead_removals}"
            ),
            fill="black",
        )
        draw.text(
            (MARGIN, 100),
            f"Last completed action: {snapshot.last_action_text}",
            fill="black",
        )

        stage_symbol = {
            "SEED": "S",
            "GROWING": "G",
            "HARVESTABLE": "H",
            "HARVESTED": "h",
            "DEAD": "D",
        }

        for floor in range(farm.floors):
            x_origin = MARGIN + floor * (floor_width + FLOOR_GAP)
            y_origin = HEADER
            draw.text((x_origin, y_origin - 18), f"Floor {floor}", fill="black")

            for row in range(farm.rows):
                for column in range(farm.columns):
                    position = (floor, row, column)
                    state = snapshot.plants[position]
                    x1 = x_origin + column * CELL
                    y1 = y_origin + row * CELL
                    x2 = x1 + CELL
                    y2 = y1 + CELL

                    draw.rectangle((x1, y1, x2, y2), outline="black", width=1)
                    if state.has_request:
                        draw.rectangle(
                            (x1 + 3, y1 + 3, x2 - 3, y2 - 3),
                            outline="black",
                            width=3,
                        )

                    species_abbreviation = state.species_name[:2].upper()
                    stage = stage_symbol.get(state.growth_stage, "?")
                    draw.text(
                        (x1 + 6, y1 + 7),
                        f"{species_abbreviation}:{stage}",
                        fill="black",
                    )

                    if snapshot.current_task_position == position:
                        draw.text((x1 + CELL - 17, y1 + 4), "*", fill="black")

            for plan_number, position in enumerate(snapshot.lookahead_plan, start=1):
                if position[0] == floor:
                    _, row, column = position
                    x = x_origin + column * CELL + CELL - 18
                    y = y_origin + row * CELL + CELL - 20
                    draw.text((x, y), str(plan_number), fill="black")

            if snapshot.agent_position[0] == floor:
                _, row, column = snapshot.agent_position
                center_x = x_origin + column * CELL + CELL // 2
                center_y = y_origin + row * CELL + CELL // 2
                radius = 11
                draw.ellipse(
                    (
                        center_x - radius,
                        center_y - radius,
                        center_x + radius,
                        center_y + radius,
                    ),
                    outline="black",
                    width=4,
                )
                draw.text((center_x - 4, center_y - 8), "A", fill="black")

        draw.text(
            (MARGIN, height - 24),
            (
                "A agent | * committed task | 1..n lookahead | outlined request | "
                "species initials: S seed, G growing, H harvestable, D dead"
            ),
            fill="black",
        )
        return image
