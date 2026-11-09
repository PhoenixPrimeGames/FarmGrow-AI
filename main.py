"""Command-line entry point for configuring, running, and visualizing the farm."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Optional

from agent import FarmAgent
from farm import Farm3D
from plant_class import PlantGrowthStages
from plant_data import load_plant_species_csv
from project_info import PROJECT_NAME
from reporting import RunStatisticsRecorder
from simulation import FarmSimulation
from visualizer import GifVisualizerRecorder, SECONDS_PER_DAY


# Show the per-species share of the farm that the agent successfully harvested.
def print_harvest_efficiency(
        statistics: RunStatisticsRecorder,
        simulation: FarmSimulation,
) -> None:
    """Print harvested, dead, and outcome-efficiency values by species."""
    print("\nHarvest efficiency by species:")
    print(
        "  Efficiency = Harvested / (Harvested + Dead)\n"
        f"  {'Species':<20} {'Harvested':>10} {'Dead':>10} {'Efficiency':>12}"
    )
    for row in statistics.species_summary_rows(simulation):
        species = row["Species"]
        harvested = row["Harvested"]
        dead = row["Dead Plants"]
        percentage = row["Harvest Efficiency Percentage"]
        print(
            f"  {species:<20} {harvested:>10} {dead:>10} "
            f"{percentage:>9.2f}%"
        )


# Build the command-line parser used for optional non-interactive configuration.
def build_argument_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the vertical farm simulation."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME}: 3-D vertical farm agent simulation."
    )
    parser.add_argument("--plant-data", type=Path)
    parser.add_argument("--floors", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--species", type=str)
    parser.add_argument("--layout", choices=("cycle", "random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--start-floor", type=int)
    parser.add_argument("--start-row", type=int)
    parser.add_argument("--start-column", type=int)
    parser.add_argument("--lookahead", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--days", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--max-frames-per-gif", type=int)
    parser.add_argument("--real-seconds-per-day", type=float)
    parser.add_argument("--mode", choices=("batch", "realtime3d"))
    parser.add_argument("--3d-speed", dest="three_d_speed", type=float)
    parser.add_argument("--initial-requests", type=int)
    return parser


# Prompt repeatedly until the user enters a valid integer within fixed bounds.
def prompt_int(
        prompt: str,
        default: int,
        minimum: int,
        maximum: int,
) -> int:
    """Read and validate an integer from the user."""
    value: int = default
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()

        if raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Enter a whole number.")
                continue

        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if value > maximum:
            print(f"Value must be at most {maximum}.")
            continue
        break

    return value


# Ask the user which loaded species should be used to populate the initial farm.
def prompt_species_selection(
        available_species: list[str],
        supplied_value: Optional[str],
) -> list[str]:
    """Return validated selected species names from CLI input or an interactive prompt."""
    print("\nAvailable plant species:")
    for index, name in enumerate(available_species, start=1):
        print(f"  {index}. {name}")

    while True:
        raw = supplied_value
        if raw is None:
            raw = input(
                "\nSelect species by number or name, comma-separated "
                "[all species]: "
            ).strip()

        if raw == "":
            return available_species

        try:
            selected: list[str] = []
            for token in (part.strip() for part in raw.split(",")):
                if token.isdigit():
                    index = int(token)
                    if not 1 <= index <= len(available_species):
                        raise ValueError(
                            f"Species index out of range: {index}"
                        )
                    name = available_species[index - 1]
                else:
                    matches = [
                        name
                        for name in available_species
                        if name.casefold() == token.casefold()
                    ]
                    if not matches:
                        raise ValueError(f"Unknown species: {token}")
                    name = matches[0]

                if name not in selected:
                    selected.append(name)

            return selected
        except ValueError:
            if supplied_value is not None:
                raise
            print("Invalid species selection. Try again.")


# Resolve a command-line value or interactively ask for one of the allowed text choices.
def resolve_choice(
        supplied_value: Optional[str],
        prompt: str,
        default: str,
        allowed: set[str],
) -> str:
    """Return a validated string choice."""
    if supplied_value is not None:
        return supplied_value

    while True:
        raw = input(f"{prompt} [{default}]: ").strip().lower()
        value = default if raw == "" else raw
        if value in allowed:
            return value
        print(f"Choose one of: {', '.join(sorted(allowed))}")


# Resolve one integer from CLI input or prompt the user when it was omitted.
def resolve_int(
        supplied_value: Optional[int],
        prompt: str,
        default: int,
        minimum: int = 1,
        maximum: int = 1000,
) -> int:
    """Return a validated supplied integer or prompt for one."""
    if supplied_value is not None:
        if supplied_value < minimum:
            raise ValueError(f"{prompt} must be at least {minimum}.")
        if supplied_value > maximum:
            raise ValueError(f"{prompt} must be at most {maximum}.")
        return supplied_value

    return prompt_int(prompt, default, minimum, maximum)


# Resolve one positive floating-point value from CLI input or an interactive prompt.
def resolve_positive_float(
        supplied_value: Optional[float],
        prompt: str,
        default: float,
) -> float:
    """Return a positive float supplied on the CLI or entered interactively."""
    if supplied_value is not None:
        if not math.isfinite(supplied_value) or supplied_value <= 0:
            raise ValueError(f"{prompt} must be a finite number greater than zero.")
        return float(supplied_value)

    value: float = default
    while True:
        raw = input(f"{prompt} [{default:g}]: ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Enter a number.")
                continue

        if not math.isfinite(value) or value <= 0:
            print("Value must be a finite number greater than zero.")
            continue
        break

    return value


# Seed the requested number of plants with visible care requests.
def seed_initial_requests(
        farm: Farm3D,
        request_count: int,
        random_seed: int,
) -> None:
    """Create deterministic initial requests at distinct plant positions."""
    positions = list(farm.positions())
    if not 0 <= request_count <= len(positions):
        raise ValueError(
            "Initial request count must be between zero and the number of plants."
        )

    request_rng = random.Random(random_seed)
    selected_positions = request_rng.sample(
        positions,
        k=request_count,
    )

    for request_index, position in enumerate(selected_positions):
        request_type = request_index % 4
        if request_type == 0:
            plant = farm.get_plant(position)
            plant.current_growth_stage = PlantGrowthStages.HARVESTABLE
        elif request_type == 1:
            farm.get_plant(position).kill_plant()
        elif request_type == 2:
            plant = farm.get_plant(position)
            plant.current_ph_level = plant.species.ph_range[1] + 1.0
        else:
            plant = farm.get_plant(position)
            plant.current_ec_level = plant.species.ec_range[1] + 1.0


# Build the complete farm, agent, simulation, and recorder from user configuration.
def build_simulation_from_user_input(
        args: argparse.Namespace,
) -> tuple[FarmSimulation, GifVisualizerRecorder]:
    """Load species data and create the one authoritative simulation run."""
    script_directory = Path(__file__).resolve().parent
    plant_data_path = args.plant_data or script_directory / "plant_species.csv"
    print("Welcome to FarmGrow AI!\n"
          "Let's start by building your farm:\n")
    mode = resolve_choice(
        args.mode,
        "Run mode (batch/realtime3d)",
        "realtime3d",
        {"batch", "realtime3d"},
    )
    args.mode = mode

    species_by_name = load_plant_species_csv(plant_data_path)
    available_species = list(species_by_name)

    selected_names = prompt_species_selection(
        available_species,
        args.species,
    )
    selected_species = [
        species_by_name[name] for name in selected_names
    ]

    floors = resolve_int(
        args.floors,
        "Number of floors",
        3,
        minimum=1,
        maximum=Farm3D.MAX_PLANT_LOCATIONS,
    )
    floor_area_limit = Farm3D.MAX_PLANT_LOCATIONS // floors
    print(
        "\nCapacity after selecting floors:\n"
        f"  {floor_area_limit:,} plant locations per floor are available.\n"
        f"  Maximum rows: {floor_area_limit:,} with 1 column.\n"
        f"  Maximum columns: {floor_area_limit:,} with 1 row.\n"
        f"  Rows x columns must not exceed {floor_area_limit:,}.\n"
    )

    rows = resolve_int(
        args.rows,
        "Rows per floor",
        min(5, floor_area_limit),
        minimum=1,
        maximum=floor_area_limit,
    )
    column_limit = Farm3D.MAX_PLANT_LOCATIONS // (floors * rows)
    print(
        "\nCapacity after selecting rows:\n"
        f"  With {floors:,} floor(s) and {rows:,} row(s), you can use "
        f"up to {column_limit:,} column(s).\n"
    )
    columns = resolve_int(
        args.columns,
        "Columns per floor",
        min(6, column_limit),
        minimum=1,
        maximum=column_limit,
    )
    layout = resolve_choice(
        args.layout,
        "Species layout (cycle/random)",
        "cycle",
        {"cycle", "random"},
    )
    seed = resolve_int(args.seed, "Random seed", 1)

    start_floor = resolve_int(
        args.start_floor,
        "Agent start floor",
        0,
        minimum=0,
        maximum=floors - 1,
    )
    start_row = resolve_int(
        args.start_row,
        "Agent start row",
        0,
        minimum=0,
        maximum=rows - 1,
    )
    start_column = resolve_int(
        args.start_column,
        "Agent start column",
        0,
        minimum=0,
        maximum=columns - 1,
    )

    lookahead = resolve_int(
        args.lookahead,
        "Lookahead depth",
        3,
        minimum=1,
        maximum=FarmAgent.MAX_LOOKAHEAD_DEPTH,
    )
    candidate_limit = resolve_int(
        args.candidate_limit,
        "Maximum requests considered per replan",
        12,
        minimum=1,
        maximum=FarmAgent.MAX_PLANNING_CANDIDATES,
    )
    print(f"\n{floors * rows * columns} plants are available for initial plant requests.\n")
    initial_request_count = resolve_int(
        args.initial_requests,
        "Plants starting with a care request",
        0,
        minimum=0,
        maximum=floors * rows * columns,
    )

    default_output_directory = Path.home() / "Documents" / "FarmGrow AI"
    run_days: int
    max_frames: int
    simulated_seconds_per_real_second: float
    output_directory: Path
    if mode == "realtime3d":
        simulated_seconds_per_real_second = 1.0
        run_days = 1
        max_frames = 900
        output_directory = args.output_directory or default_output_directory
        args.three_d_speed = resolve_positive_float(
            args.three_d_speed,
            "Initial 3-D simulation speed",
            1.0,
        )
    else:
        run_days = resolve_int(
            args.days,
            "Run time in days",
            1,
            minimum=1,
        )
        max_frames = resolve_int(
            args.max_frames_per_gif,
            "Maximum frames per GIF",
            900,
            minimum=2,
        )
        real_seconds_per_day = resolve_positive_float(
            args.real_seconds_per_day,
            "Real seconds per simulated day",
            1.0,
        )
        simulated_seconds_per_real_second = (
                SECONDS_PER_DAY / real_seconds_per_day
        )
        if args.output_directory is not None:
            output_directory = args.output_directory
        else:
            raw_output_directory = input(
                f"Output directory [{default_output_directory}]: "
            ).strip()
            output_directory = (
                Path(raw_output_directory)
                if raw_output_directory
                else default_output_directory
            )

    farm = Farm3D(
        floors=floors,
        rows=rows,
        columns=columns,
        species_options=selected_species,
        layout_mode=layout,
        random_seed=seed,
    )
    seed_initial_requests(farm, initial_request_count, seed)

    agent = FarmAgent(
        farm=farm,
        start_position=(start_floor, start_row, start_column),
        lookahead_depth=lookahead,
        planning_candidate_limit=candidate_limit,
    )
    simulation = FarmSimulation(
        farm=farm,
        agent=agent,
        simulated_seconds_per_real_second=(
            simulated_seconds_per_real_second
        ),
    )
    if initial_request_count > 0:
        agent.replan(simulation.clock.total_simulated_seconds)

    recorder = GifVisualizerRecorder(
        total_run_days=run_days,
        output_directory=output_directory,
        max_frames_per_gif=max_frames,
    )

    return simulation, recorder


# Run the configured simulation once and generate GIFs from snapshots of that exact run.
def main() -> None:
    """Configure one simulation, run it, and create 30-day GIF chunks."""
    parser = build_argument_parser()
    args = parser.parse_args()

    simulation, recorder = build_simulation_from_user_input(args)
    statistics = RunStatisticsRecorder(simulation)
    simulation.add_observer(statistics.observe)

    if args.mode == "realtime3d":
        three_d_speed = args.three_d_speed
        if three_d_speed is None:
            raise RuntimeError("The validated 3-D speed is missing.")

        from realtime_3d import Realtime3DRenderer, Realtime3DSettings

        renderer = Realtime3DRenderer(
            simulation,
            Realtime3DSettings(
                simulation_speed=three_d_speed,
            ),
        )
        print("\nStarting interactive 3-D simulation...")
        print(f"Observation speed: {three_d_speed:g}x")
        renderer.run()
        return

    recorder.capture_initial_state(simulation)
    simulation.add_observer(recorder.observe)

    print("\nRunning simulation...")
    print(
        "Time compression: "
        f"{simulation.clock.simulated_seconds_per_real_second:.2f} "
        "simulated seconds per real second"
    )
    simulation.run_realtime(recorder.total_run_days)

    print("Creating GIF visualization files from the completed run...")
    gif_paths = recorder.write_gifs(simulation)
    report_directory = recorder.output_directory / "reports"
    report_paths = statistics.write_csv_reports(simulation, report_directory)

    summary = simulation.summary()
    print("\nSimulation complete")
    print(f"Simulated days: {recorder.total_run_days}")
    print(f"Agent position: {summary['agent_position']}")
    print(f"Distance traveled: {summary['distance_traveled']}")
    print(f"Completed tasks: {summary['completed_tasks']}")
    print(f"Completed actions: {summary['completed_actions']}")

    print("\nFarm species counts:")
    for species_name, count in simulation.farm.species_counts().items():
        print(f"  {species_name}: {count}")

    print_harvest_efficiency(statistics, simulation)

    print("\nCreated GIF files:")
    for path in gif_paths:
        print(f"  {path}")

    print("\nCreated report files:")
    for path in report_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
