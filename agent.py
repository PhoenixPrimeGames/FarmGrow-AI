"""Receding-horizon service agent for the three-dimensional farm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from typing import Optional, TypedDict

from farm import Farm3D, FarmEvent, FarmRequest, Position
from plant_class import PlantGrowthStages, PlantRequest


class ServiceAction(Enum):
    """Physical service actions with action-specific simulated durations."""

    HARVEST = "HARVEST"
    ADJUST_PH = "ADJUST_PH"
    REPLENISH_NUTRIENTS = "REPLENISH_NUTRIENTS"
    REMOVE_DEAD = "REMOVE_DEAD"
    PLANT = "PLANT"


@dataclass
class PlantTask:
    """The agent-ready service work required at one plant location."""

    position: Position
    actions: list[ServiceAction]
    priority: float

    # Compute the total service time using the agent's action-duration settings.
    def service_seconds(self, action_durations: dict[ServiceAction, int]) -> int:
        """Return the seconds required to finish every action in this task."""
        return sum(action_durations[action] for action in self.actions)


class ActionRecord(TypedDict):
    """One completed service action retained for reporting."""

    elapsed_second: int
    position: Position
    species: str
    action: str


class FarmAgent:
    """A depth-limited receding-horizon agent with timed movement and service actions."""

    HORIZONTAL_MOVE_SECONDS = 2
    VERTICAL_MOVE_SECONDS = 4
    MAX_LOOKAHEAD_DEPTH = 3
    MAX_PLANNING_CANDIDATES = 12

    ACTION_DURATIONS = {
        ServiceAction.HARVEST: 10,
        ServiceAction.ADJUST_PH: 10,
        ServiceAction.REPLENISH_NUTRIENTS: 15,
        ServiceAction.REMOVE_DEAD: 15,
        ServiceAction.PLANT: 20,
    }

    # Initialize the agent's position, planning limits, and runtime state.
    def __init__(
            self,
            farm: Farm3D,
            start_position: Position = (0, 0, 0),
            lookahead_depth: int = 3,
            planning_candidate_limit: int = 12,
    ) -> None:
        """Create an agent attached to a farm environment."""
        if not farm.in_bounds(start_position):
            raise ValueError("Agent start position must be inside the farm.")
        if lookahead_depth <= 0:
            raise ValueError("Lookahead depth must be greater than zero.")
        if lookahead_depth > self.MAX_LOOKAHEAD_DEPTH:
            raise ValueError(
                f"Lookahead depth cannot exceed {self.MAX_LOOKAHEAD_DEPTH}."
            )
        if planning_candidate_limit <= 0:
            raise ValueError("Planning candidate limit must be greater than zero.")
        if planning_candidate_limit > self.MAX_PLANNING_CANDIDATES:
            raise ValueError(
                "Planning candidate limit cannot exceed "
                f"{self.MAX_PLANNING_CANDIDATES}."
            )

        self.farm = farm
        self.position = start_position
        self.lookahead_depth = lookahead_depth
        self.planning_candidate_limit = planning_candidate_limit

        self.current_task: Optional[PlantTask] = None
        self.current_action_index = 0
        self.action_seconds_remaining = 0
        self.movement_path: list[Position] = []
        self.next_move_position: Optional[Position] = None
        self.move_seconds_remaining = 0
        self.current_move_duration = 0

        self.lookahead_plan: list[Position] = []
        self.replan_pending = False

        self.completed_tasks = 0
        self.completed_actions = 0
        self.distance_traveled = 0
        self.event_log: list[str] = []
        self.action_records: list[ActionRecord] = []

    @staticmethod
    # Calculate obstacle-free three-dimensional Manhattan travel distance.
    def manhattan_distance(a: Position, b: Optional[Position]) -> int:
        """Return the number of adjacent axis-aligned steps between two positions."""
        if b is None:
            return 0
        return sum(abs(x - y) for x, y in zip(a, b))

    @classmethod
    # Calculate simulated travel time using separate horizontal and vertical movement speeds.
    def travel_seconds(cls, a: Position, b: Position) -> int:
        """Return simulated seconds needed to travel between two farm positions."""
        floor_a, row_a, column_a = a
        floor_b, row_b, column_b = b
        horizontal_steps = abs(row_a - row_b) + abs(column_a - column_b)
        vertical_steps = abs(floor_a - floor_b)
        return (
                horizontal_steps * cls.HORIZONTAL_MOVE_SECONDS
                + vertical_steps * cls.VERTICAL_MOVE_SECONDS
        )

    @staticmethod
    # Generate a deterministic adjacent-cell movement path between two coordinates.
    def manhattan_path(start: Position, goal: Position) -> list[Position]:
        """Return adjacent positions from start-exclusive to goal-inclusive."""
        floor, row, column = start
        goal_floor, goal_row, goal_column = goal
        path: list[Position] = []

        while column != goal_column:
            column += 1 if goal_column > column else -1
            path.append((floor, row, column))

        while row != goal_row:
            row += 1 if goal_row > row else -1
            path.append((floor, row, column))

        while floor != goal_floor:
            floor += 1 if goal_floor > floor else -1
            path.append((floor, row, column))

        return path

    # Expand the current commitment and lookahead into exact adjacent farm cells.
    def planned_route_cells(self) -> list[Position]:
        """Return the axis-aligned cells the currently displayed plan will use."""
        task = self.current_task
        if task is None:
            return []

        cells: list[Position] = []
        current = self.position
        if self.next_move_position is not None:
            cells.append(self.next_move_position)
            current = self.next_move_position

        # A changed target will be reconsidered as soon as this active step ends.
        if self.replan_pending and self.next_move_position is not None:
            return cells

        for position in self.movement_path:
            if position != current:
                cells.append(position)
                current = position

        if current != task.position:
            segment = self.manhattan_path(current, task.position)
            cells.extend(segment)
            current = task.position

        for position in self.lookahead_plan[1:]:
            segment = self.manhattan_path(current, position)
            cells.extend(segment)
            current = position

        return cells

    @property
    # Report the discrete cell from which the current movement segment began.
    def movement_start_position(self) -> Position:
        """Return the current discrete grid cell used as the movement segment origin."""
        return self.position

    @property
    # Report the adjacent cell toward which the agent is currently moving.
    def movement_target_position(self) -> Optional[Position]:
        """Return the active adjacent movement target, or None when not between cells."""
        return self.next_move_position

    @property
    # Report normalized completion of the active movement segment for rendering.
    def movement_progress(self) -> float:
        """Return movement completion from 0.0 at departure to 1.0 at arrival."""
        if self.next_move_position is None or self.current_move_duration <= 0:
            return 0.0

        elapsed = self.current_move_duration - self.move_seconds_remaining
        return max(0.0, min(1.0, elapsed / self.current_move_duration))

    # Compute a continuous renderer position with optional sub-second interpolation.
    def render_position_at(
            self,
            fractional_simulated_second: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return a continuous (floor, row, column) position for smooth visualization."""
        start = self.movement_start_position
        target = self.movement_target_position

        if target is None or self.current_move_duration <= 0:
            return start

        base_elapsed = (
                self.current_move_duration - self.move_seconds_remaining
        )
        elapsed = base_elapsed + max(
            0.0,
            min(0.999999, fractional_simulated_second),
        )
        progress = max(
            0.0,
            min(1.0, elapsed / self.current_move_duration),
        )
        ret_value: tuple[float, float, float] = (
            float(start[0] + (target[0] - start[0]) * progress),
            float(start[1] + (target[1] - start[1]) * progress),
            float(start[2] + (target[2] - start[2]) * progress),
        )
        return ret_value

    @property
    # Report the current whole-tick continuous position for non-interactive consumers.
    def render_position(self) -> tuple[float, float, float]:
        """Return the continuous render position using completed simulation ticks."""
        return self.render_position_at(0.0)

    @property
    # Report the agent's current high-level behavior for logging and visualization.
    def current_action_name(self) -> str:
        """Return IDLE, MOVING, or the current service action name."""
        task = self.current_task
        if task is None:
            return "IDLE"
        if self.position != task.position:
            return "MOVING"
        if self.current_action_index < len(task.actions):
            return task.actions[self.current_action_index].value
        return "FINISHING"

    # React only to actionable farm request transitions instead of every minute.
    def on_farm_events(
            self,
            events: list[FarmEvent],
            current_second: int,
    ) -> None:
        """Wake when idle or refresh a changed target at a safe boundary."""
        if not events:
            return

        task = self.current_task
        if task is None:
            if any(event.current_actions for event in events):
                self.event_log.append(
                    f"{current_second:08d}s FARM_EVENT wake_idle"
                )
                self.replan(current_second)
            return

        target_changed = any(
            event.position == task.position for event in events
        )
        if not target_changed:
            self.event_log.append(
                f"{current_second:08d}s FARM_EVENT retain_target={task.position}"
            )
            return

        self.replan_pending = True
        if self.position != task.position:
            self.event_log.append(
                f"{current_second:08d}s FARM_EVENT target_changed "
                "defer_until_move_done"
            )
        else:
            self.event_log.append(
                f"{current_second:08d}s FARM_EVENT target_changed "
                "defer_until_task_complete"
            )

    # Provide a low-frequency fallback in case an external mutation emitted no event.
    def on_safety_check(self, current_second: int) -> None:
        """Rescan only when idle during the daily safety check."""
        if self.current_task is None:
            self.event_log.append(
                f"{current_second:08d}s SAFETY_CHECK idle_replan"
            )
            self.replan(current_second)

    # Convert farm requests to agent tasks, search three tasks ahead, and commit only to the first.
    def replan(self, current_second: int) -> None:
        """Calculate a fresh lookahead plan and commit to only its first task."""
        requests = self.farm.collect_requests()
        tasks = [self._request_to_task(request) for request in requests]

        self.lookahead_plan = self.plan_next_tasks(tasks, self.position)
        self.replan_pending = False
        self.event_log.append(
            f"{current_second:08d}s REPLAN lookahead={self.lookahead_plan}"
        )

        self.current_task = None
        self.current_action_index = 0
        self.action_seconds_remaining = 0
        self.movement_path = []
        self.next_move_position = None
        self.move_seconds_remaining = 0
        self.current_move_duration = 0

        if self.lookahead_plan:
            self._commit_to_position(self.lookahead_plan[0], current_second)

    # Evaluate ordered task routes up to the configured lookahead depth.
    def plan_next_tasks(
            self,
            tasks: list[PlantTask],
            start: Position,
    ) -> list[Position]:
        """Return the best depth-limited route according to priority and elapsed time."""
        if not tasks:
            return []

        ranked_tasks = sorted(
            tasks,
            key=lambda _task: (
                -_task.priority,
                self.travel_seconds(start, _task.position),
                _task.position,
            ),
        )
        candidate_tasks = ranked_tasks[: self.planning_candidate_limit]
        depth = min(self.lookahead_depth, len(candidate_tasks))

        best_route: Optional[tuple[PlantTask, ...]] = None
        best_key: Optional[
            tuple[float, float, float, tuple[Position, ...]]
        ] = None

        for route in permutations(candidate_tasks, depth):
            current = start
            elapsed = 0
            weighted_completion = 0.0
            total_priority = 0.0

            for task in route:
                elapsed += self.travel_seconds(current, task.position)
                elapsed += task.service_seconds(self.ACTION_DURATIONS)
                total_priority += task.priority
                weighted_completion += (task.priority + 1.0) * elapsed
                current = task.position

            positions = tuple(task.position for task in route)
            key = (
                -total_priority,
                weighted_completion,
                float(elapsed),
                positions,
            )

            if best_key is None or key < best_key:
                best_key = key
                best_route = route

        return [task.position for task in best_route] if best_route else []

    # Advance either movement or servicing by exactly one simulated second.
    def update_one_second(self, current_second: int) -> None:
        """Perform one second of the agent's currently committed behavior."""
        task = self.current_task
        if task is None:
            return

        if self.position != task.position:
            self._move_one_second(current_second)
            return

        self._service_one_second(current_second)

    # Translate one farm-level request snapshot into the physical actions the agent must perform.
    @staticmethod
    def _request_to_task(request: FarmRequest) -> PlantTask:
        """Convert a FarmRequest into an ordered PlantTask."""
        plant = request.plant

        if plant.current_growth_stage == PlantGrowthStages.DEAD:
            actions = [ServiceAction.REMOVE_DEAD, ServiceAction.PLANT]
        elif plant.current_growth_stage == PlantGrowthStages.HARVESTED:
            actions = [ServiceAction.PLANT]
        elif PlantRequest.HARVEST in request.requested_actions:
            actions = [ServiceAction.HARVEST, ServiceAction.PLANT]
        else:
            actions = []
            if PlantRequest.ADJUST_PH in request.requested_actions:
                actions.append(ServiceAction.ADJUST_PH)
            if PlantRequest.REPLENISH_NUTRIENTS in request.requested_actions:
                actions.append(ServiceAction.REPLENISH_NUTRIENTS)

        return PlantTask(
            position=request.position,
            actions=actions,
            priority=request.priority,
        )

    # Commit to the current version of a request at a specific position.
    def _commit_to_position(self, position: Position, current_second: int) -> None:
        """Set the current task and movement path for the selected position."""
        request = next(
            (
                request
                for request in self.farm.collect_requests()
                if request.position == position
            ),
            None,
        )

        if request is None:
            self.event_log.append(
                f"{current_second:08d}s COMMIT_SKIPPED stale_request={position}"
            )
            self.replan(current_second)
            return

        task = self._request_to_task(request)
        self.current_task = task
        self.current_action_index = 0
        self.action_seconds_remaining = 0
        self.movement_path = self.manhattan_path(self.position, position)
        self.next_move_position = None
        self.move_seconds_remaining = 0
        self.current_move_duration = 0
        self.event_log.append(
            f"{current_second:08d}s COMMIT {position} "
            f"actions={[action.value for action in task.actions]}"
        )

    # Spend simulated time moving toward the next adjacent grid cell.
    def _move_one_second(self, current_second: int) -> None:
        """Consume one second of movement and enter the next cell only when travel time finishes."""
        task = self.current_task
        if task is None:
            return

        next_position = self.next_move_position
        if next_position is None:
            if not self.movement_path:
                self.movement_path = self.manhattan_path(
                    self.position,
                    task.position,
                )

            next_position = self.movement_path.pop(0)
            self.next_move_position = next_position

            if self.manhattan_distance(self.position, next_position) != 1:
                raise RuntimeError("Movement path attempted a non-adjacent move.")
            if not self.farm.in_bounds(next_position):
                raise RuntimeError("Movement path left the farm bounds.")

            if next_position[0] != self.position[0]:
                self.current_move_duration = self.VERTICAL_MOVE_SECONDS
            else:
                self.current_move_duration = self.HORIZONTAL_MOVE_SECONDS

            self.move_seconds_remaining = self.current_move_duration

            self.event_log.append(
                f"{current_second:08d}s MOVE_START "
                f"{self.position}->{next_position} "
                f"duration={self.move_seconds_remaining}s"
            )

        self.move_seconds_remaining -= 1

        if self.move_seconds_remaining == 0:
            self.position = next_position
            self.next_move_position = None
            self.current_move_duration = 0
            self.distance_traveled += 1
            self.event_log.append(
                f"{current_second:08d}s MOVE_DONE {self.position}"
            )
            if self.replan_pending:
                self.event_log.append(
                    f"{current_second:08d}s FARM_EVENT replan_after_move"
                )
                self.replan(current_second)

    # Spend one second on the current service action and complete actions at five-second intervals.
    def _service_one_second(self, current_second: int) -> None:
        """Consume one second of service work at the current plant."""
        task = self.current_task
        if task is None:
            return

        if self.current_action_index >= len(task.actions):
            self._finish_current_task(current_second)
            return

        if self.action_seconds_remaining == 0:
            action = task.actions[self.current_action_index]
            self.action_seconds_remaining = self.ACTION_DURATIONS[action]
            self.event_log.append(
                f"{current_second:08d}s START {action.value} "
                f"at {task.position}"
            )

        self.action_seconds_remaining -= 1

        if self.action_seconds_remaining == 0:
            action = task.actions[self.current_action_index]
            plant = self.farm.get_plant(task.position)
            species_name = plant.species.name
            self._apply_action(action, task)
            self.completed_actions += 1
            self.action_records.append(
                {
                    "elapsed_second": current_second,
                    "position": task.position,
                    "species": species_name,
                    "action": action.value,
                }
            )
            self.event_log.append(
                f"{current_second:08d}s DONE {action.value} "
                f"{species_name} at {task.position}"
            )
            self.current_action_index += 1

            if self.current_action_index >= len(task.actions):
                self._finish_current_task(current_second)

    # Apply the completed physical service action to the plant stored in the farm.
    def _apply_action(self, action: ServiceAction, task: PlantTask) -> None:
        """Mutate the target plant according to a completed service action."""
        plant = self.farm.get_plant(task.position)

        if action == ServiceAction.HARVEST:
            if not plant.harvest_plant():
                raise RuntimeError(f"Could not harvest plant at {task.position}.")
        elif action == ServiceAction.ADJUST_PH:
            if not plant.adjust_ph():
                raise RuntimeError(f"Could not adjust pH at {task.position}.")
        elif action == ServiceAction.REPLENISH_NUTRIENTS:
            if not plant.replenish_nutrients():
                raise RuntimeError(
                    f"Could not replenish nutrients at {task.position}."
                )
        elif action == ServiceAction.REMOVE_DEAD:
            if plant.current_growth_stage != PlantGrowthStages.DEAD:
                raise RuntimeError(
                    f"Plant at {task.position} is not dead and cannot be removed."
                )
        elif action == ServiceAction.PLANT:
            if not plant.replace_plant(plant.species):
                raise RuntimeError(f"Could not plant at {task.position}.")
        else:
            raise ValueError(f"Unsupported service action: {action}")

    # Finish the full plant task and immediately calculate a fresh three-task lookahead.
    def _finish_current_task(self, current_second: int) -> None:
        """Close the committed task and immediately replan from the current position."""
        task = self.current_task
        if task is None:
            return

        finished_position = task.position
        self.completed_tasks += 1
        self.event_log.append(
            f"{current_second:08d}s TASK_DONE {finished_position}"
        )

        self.current_task = None
        self.current_action_index = 0
        self.action_seconds_remaining = 0
        self.movement_path = []
        self.next_move_position = None
        self.move_seconds_remaining = 0
        self.current_move_duration = 0

        self.replan(current_second)
