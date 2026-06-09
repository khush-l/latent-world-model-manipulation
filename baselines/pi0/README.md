# pi0 / OpenPI Baseline

This folder keeps the pi0 LoRA baseline code for SoftGym RopeFlatten. It is
kept separate from the ACT/SmolVLA helpers because OpenPI uses different data
keys and a different policy runtime.

## Files

| File | Purpose |
|---|---|
| `softgym_hdf5_to_openpi_lerobot.py` | Convert SoftGym HDF5 data to the LeRobot layout expected by the OpenPI transform. |
| `install_openpi_softgym_config.sh` | Copy/install the SoftGym OpenPI config into the OpenPI checkout. |
| `train_pi0_lora_softgym_rope.sh` | Run pi0 LoRA training on the converted RopeFlatten dataset. |
| `openpi_softgym_server.py` | Serve a trained pi0 policy with the same HTTP API used by the SoftGym eval client. |
| `openpi_softgym_policy.py` | SoftGym action adapter for OpenPI policy outputs. |
| `eval_pi0_lora_softgym_rope.sh` | Start the pi0 policy server and run the SoftGym eval client. |

## Notes

The action adapter clips OpenPI outputs into the 8D SoftGym two-picker action
space. Evaluation reports action chunk latency separately from cached-action
latency, which is the fairer number for direct-action policies.
