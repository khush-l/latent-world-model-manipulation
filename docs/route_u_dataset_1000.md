# Route-U RopeConfiguration Dataset

Dataset:
`simulation/data/ropeconfiguration_route_u_1000_collectonly_060_20260602_112213.h5`

Purpose:
LeWM/JEPA training data for a practical U-shaped rope/cable-routing task, scoped as the robust complex RopeConfiguration task.

Collection settings:
- Policy: `route_u`
- Goal: `U`
- Episodes: 1000
- Shards: 10 x 100 episodes
- Image size: 128
- Horizon: 300
- Action noise: 0.0
- Acceptance gate: final `info_normalized_performance >= 0.60`
- Minimum accepted interaction length: 20 actions

Validation summary:
- NPZ episodes collected: 1000
- Merged HDF5 episodes: 1000
- HDF5 rows: 89,760
- Bad accepted episodes: 0
- Final normalized performance: min 0.6000, mean 0.6303, median 0.6202, p95 0.7020, max 0.7624
- Episode length: min 23, mean 88.76, median 65, p95 242, max 300
- Unique reset configs: 108
- Pixel range: 0..255, nonblank std 20.3540
- Terminal action/reward padding: valid NaN padding at terminal rows
- `/stats/{action,proprio,state}` present

Preview:
`simulation/data/ropeconfiguration_route_u_1000_collectonly_060_20260602_112213/route_u_1000_final_preview.png`
