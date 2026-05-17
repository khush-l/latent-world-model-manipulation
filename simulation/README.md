This folder contains simulation environment scripts and tools, including SoftGym base code built on NVIDIA FleX.

Utilities:

- `utils/collect_trajectories.py`: collect SoftGym trajectories into `.npz` files.
- `utils/view_trajectory.py`: generate a static HTML viewer for collected trajectories.
- `softagent/`: vendored SoftAgent benchmark algorithms for CEM, CURL/SAC, DrQ, PlaNet, and MVP.

Run a quick SoftAgent CEM simulation through Docker:

```bash
docker/softagent-local.sh cem --env-name ClothFlatten --test-episodes 1 --max-iters 2 --timestep-per-decision 200
```
