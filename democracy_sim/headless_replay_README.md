# Headless Simulation and Replay Guide for DemocracySim

This guide explains how to run DemocracySim simulations in headless mode (without visualization) 
and then replay them using Mesa's visualization tools.

## Overview

DemocracySim supports two main modes of operation:

1. **Interactive Mode**: Run the simulation with real-time visualization using `run.py`
2. **Headless Mode**: Run the simulation without visualization using `run_headless.py`, saving the results to CSV files
3. **Replay Mode**: Visualize previously saved simulation data using `run_replay.py`

This workflow allows you to:
- Run computationally intensive simulations without the overhead of visualization
- Save simulation results for later analysis
- Replay simulations to visualize the dynamics and outcomes

## Running a Headless Simulation

To run a simulation in headless mode:

```bash
cd democracy_sim
python run_headless.py
```

By default, this will:
1. Load configuration from `config/config.yaml`
2. Run the simulation for the specified number of steps
3. Save the results to CSV files in the `simulation_output` directory

### Command-line Options

You can specify a custom configuration file:

```bash
python run_headless.py --config path/to/your/config.yaml
```

### Output Files

The headless simulation produces two main output files:

1. `model_data.csv`: Contains model-level data for each step, including:
   - Collective assets
   - Gini Index
   - Voter turnout
   - Color distributions
   - Grid state (as a serialized list of lists)

2. `agent_data.csv`: Contains agent-level data for each step

## Replaying a Simulation

To replay a previously saved simulation:

```bash
cd democracy_sim
python run_replay.py
```

This will:
1. Load the simulation data from `simulation_output/model_data.csv`
2. Launch a Mesa server with the same visualization elements as the interactive mode
3. Allow you to step through the simulation or play it automatically

The replay will look identical to running the simulation via `run.py`, but instead of computing the simulation dynamics in real-time, it's loading the pre-computed state from the CSV file.

### How Replay Works

The replay functionality works by:
1. Loading the fixed parameters from `config/config.yaml`
2. Loading the dynamic state data from `model_data.csv`
3. Creating a special `ReplayParticipationModel` that inherits from the regular `ParticipationModel`
4. Overriding the `step()` method to update the model state from the saved data instead of computing it
5. Using the same visualization elements as the interactive mode

## Troubleshooting

If you encounter issues with the replay:

1. **Missing GridColors column**: Ensure your `model_data.csv` file includes a `GridColors` column. This column contains the serialized grid state needed for visualization.

2. **Visualization differences**: If the replay looks different from the interactive mode, check that your visualization elements are compatible with the replay model.

3. **File not found errors**: Make sure the paths to the CSV files are correct. By default, they should be in the `simulation_output` directory.

## Advanced Usage

### Custom Data Collection

If you want to collect additional data during the headless simulation, you can modify the `initialize_datacollector` method in `participation_model.py` to include additional model reporters or agent reporters.

### Custom Replay Visualization

If you want to customize the replay visualization, you can modify `run_replay.py` to include different visualization elements or to change the appearance of the existing elements.

### Parameter Sweeps

For running multiple simulations with different parameters, consider using `parameter_sweep.py` which can run multiple headless simulations and analyze the results.