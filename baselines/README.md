# Baselines

This folder contains the cleaned baseline code for ACT, SmolVLA, and pi0 on
SoftGym RopeFlatten. The focus is on the code needed to convert data, train the
policies, run evaluation, and make result figures.

## Core Files

| File | Purpose |
|---|---|
| `softgym_hdf5_to_lerobot.py` | Convert SoftGym HDF5 rollouts into a LeRobot dataset for ACT/SmolVLA. |
| `policy_server.py` | Host-side ACT/SmolVLA policy server used by the SoftGym Docker eval client. |
| `train_rope_cem_act_after_smolvla.sh` | ACT training command for the final RopeFlatten dataset. |
| `train_rope_cem_smolvla_stronger.sh` | SmolVLA training command for the final RopeFlatten dataset. |
| `plot_lerobot_metrics.py` | Parse LeRobot training logs and plot loss/timing curves. |
| `plot_same_machine_paper_charts.py` | Build the final same-machine comparison table and figures. |
| `pi0/` | OpenPI/pi0 conversion, training, server, and eval code. |
| `results/` | Small result tables and figures. Raw videos/checkpoints/logs stay local. |

## Typical Flow

1. Convert the SoftGym HDF5 data with `softgym_hdf5_to_lerobot.py`.
2. Train ACT or SmolVLA with the matching train script.
3. Run SoftGym evaluation through `policy_server.py` and `simulation/utils/eval_lerobot_policy.py`.
4. Summarize results with `plot_same_machine_paper_charts.py`.

The older queue, tmux, handoff, and smoke-test scripts were removed from this
branch because they were only run orchestration, not core project code.
