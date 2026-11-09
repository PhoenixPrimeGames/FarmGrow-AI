# FarmGrow AI
### Automated Vertical Farm AI Simulator

A Python-based simulation of an automated vertical farm. The program models plants
in a configurable three-dimensional grid and uses a service agent to plan work
such as harvesting, adjusting pH, replenishing nutrients, and replacing plants.

Originally developed as a CPSC 4370 course project. This repository contains the
source code and configuration data, rather than an exported executable or Python
virtual environment.

## Features

- Configurable farm dimensions, plant species, initial requests, and random seed
- Plant growth stages and environmental care requirements
- Agent task prioritization and limited-lookahead, receding-horizon planning
- Batch simulations with animated GIF visualizations and CSV reports
- An interactive 3D visualization mode using Matplotlib

## Technology

Python 3.9-compatible source code, NumPy, Matplotlib, and Pillow.
See `requirements.txt` for the project's version constraints.

## Install

Use a compatible Python version (the original environment used Python 3.9).
From the repository root, create and activate a virtual environment:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On systems without the Windows Python launcher, use the appropriate `python`
or `python3` command to create the environment.

## Run

For interactive setup, start the entry point:

```bash
python main.py
```

The program prompts for the simulation mode and farm settings.
The default run mode is `realtime3d`, which uses an interactive Matplotlib window.
Choose `batch` to generate GIF visualizations and CSV reports instead.

For a small, non-interactive batch configuration:

```bash
python main.py --mode batch --species Lettuce --floors 1 --rows 2 --columns 2 --layout cycle --seed 1 --start-floor 0 --start-row 0 --start-column 0 --lookahead 2 --candidate-limit 3 --initial-requests 2 --days 1 --real-seconds-per-day 0.1 --max-frames-per-gif 12 --output-directory output
```

Batch results are written to the specified output directory. The project uses
`plant_species.csv` beside `main.py` by default.

Run `python main.py --help` for the full set of available options.

## Source files

| File | Purpose |
| --- | --- |
| `main.py` | Configuration and command-line entry point |
| `farm.py` | Three-dimensional farm environment |
| `plant_class.py` | Plant species and lifecycle behavior |
| `plant_data.py`, `plant_species.csv` | Plant-species data loading |
| `agent.py` | Agent actions, planning, and task prioritization |
| `simulation.py`, `simulation_clock.py` | Simulation progression and time |
| `reporting.py` | Statistics and CSV exports |
| `visualizer.py` | Animated GIF visualization |
| `realtime_3d.py` | Interactive 3D visualization |
| `project_info.py` | Shared application name |
| `FarmGrow AI.spec` | PyInstaller build specification, retained for reference |

## Repository scope

The original project archive included a Windows `.venv`, static-analysis caches,
and generated PyInstaller build files. These are excluded because they are
rebuildable and substantially larger than the source code.

The project currently has no committed automated test suite. Its simulation
results and behavior should be independently verified before making performance
or accuracy claims in a public portfolio.

## Portfolio context

This is an academic AI/simulation project demonstrating Python programming,
agent-based planning, data modeling, and visualization. It is a simulation,
not an independently validated agricultural optimization system.
