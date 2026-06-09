# CS231N Final Project: Rope Manipulation Baselines

This branch contains the baseline code and result summaries for the
ACT, SmolVLA, and pi0 direct-action policies used in the RopeFlatten comparison.

## What Is In This Branch

The remaining baseline files are the core pieces needed to reproduce or inspect
the baseline work:

| File | Purpose |
|---|---|
| `baselines/softgym_hdf5_to_lerobot.py` | Converts SoftGym HDF5 trajectories into LeRobot format for ACT/SmolVLA. |
| `baselines/policy_server.py` | Runs ACT/SmolVLA policies on the host and serves actions to the SoftGym Docker eval client. |
| `baselines/train_rope_cem_act_after_smolvla.sh` | ACT training command for the final RopeFlatten dataset. |
| `baselines/train_rope_cem_smolvla_stronger.sh` | SmolVLA training command for the final RopeFlatten dataset. |
| `baselines/plot_lerobot_metrics.py` | Parses LeRobot training logs and plots loss/timing curves. |
| `baselines/plot_same_machine_paper_charts.py` | Generates the final same-machine comparison table and figures. |
| `baselines/pi0/softgym_hdf5_to_openpi_lerobot.py` | Converts SoftGym data to the OpenPI-friendly LeRobot layout. |
| `baselines/pi0/train_pi0_lora_softgym_rope.sh` | Runs pi0 LoRA training. |
| `baselines/pi0/openpi_softgym_server.py` | Serves a pi0 policy with the SoftGym HTTP policy API. |
| `baselines/pi0/eval_pi0_lora_softgym_rope.sh` | Starts the pi0 server and runs SoftGym evaluation. |
| `simulation/utils/eval_lerobot_policy.py` | Docker-side SoftGym rollout evaluator for host-served policies. |
| `simulation/utils/filter_successful_trajectories.py` | Filters SoftGym trajectory files by final task score. |

## Result Summary

The final same-machine RopeFlatten comparison used 50 fixed SoftGym configs and
a 75-step horizon. Latency is reported as action-chunk latency for direct-action
policies and CEM replanning latency for LeWM+CEM.

| Method | Episodes | Mean final perf. | Latency note |
|---|---:|---:|---|
| LeWM+CEM | 50 | 0.848 | 821.6 ms CEM replan |
| ACT | 50 | 0.542 | 25.7 ms action chunk |
| SmolVLA | 50 | 0.444 | 282.6 ms action chunk |
| pi0 LoRA | 50 | 0.432 | 308.9 ms action chunk |

The final figure/table artifacts are in `baselines/results/paper_comparison/`.

Earlier baseline runs:

| Run | Episodes | Mean final perf. | Notes |
|---|---:|---:|---|
| ACT, geometric rope demos | 10 | 0.404 | early LeRobot baseline |
| SmolVLA, geometric rope demos | 10 | 0.646 | strongest early direct-action result |
| ACT, full mixed rope data | 10 | 0.400 | mixed data did not help ACT here |
| SmolVLA, full mixed rope data | 10 | 0.436 | lower than geometric-only run |
| pi0 LoRA, geometric rope demos | 10 | 0.460 | first OpenPI/pi0 eval run |

The CEM-final checkpoint sweep helped choose the longer-eval checkpoints:

| Family | Best step in sweep | Episodes | Mean final perf. |
|---|---:|---:|---:|
| ACT | 10000 | 8 | 0.548 |
| SmolVLA | 45000 | 8 | 0.493 |


## Generative AI Usage Notes

Generative AI tools were used in an assistive capacity for parts of this branch,
including code drafting/debugging, plotting cleanup, documentation cleanup, and
repository organization. Project authors reviewed the resulting code and docs,
and relevant syntax checks or figure-regeneration checks were run before commits. A best-effort record of AI-assisted work is in `ai_usage/`. 
