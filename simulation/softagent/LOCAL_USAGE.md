# Local SoftAgent Usage

This directory vendors SoftAgent from https://github.com/Xingyu-Lin/softagent.
See `ATTRIBUTION.md` for source attribution.

The upstream README assumes a separate SoftAgent checkout with `softgym/` copied
inside it. In this repository, SoftAgent is already vendored under
`simulation/softagent/`, and the Docker launcher sets `PYTHONPATH` so it uses
the sibling `simulation/softgym/` and `simulation/PyFlex/` directories.

## Build

From `simulation/`:

```bash
docker/softagent-local.sh build
```

This builds a SoftAgent image on top of the local SoftGym image and installs
PyTorch plus the benchmark-specific dependencies.

## Quick CEM Simulation

```bash
docker/softagent-local.sh cem \
  --env-name ClothFlatten \
  --test-episodes 1 \
  --max-iters 2 \
  --timestep-per-decision 200
```

Logs are written under `simulation/data/softagent/cem/`.

## Other Algorithms

The wrapper supports the benchmark families from the SoftGym paper:

```bash
docker/softagent-local.sh curl --env-name ClothFlatten --num-train-steps 10000
docker/softagent-local.sh drq --env-name ClothFlatten --num-train-steps 10000 --num-seed-steps 1000
docker/softagent-local.sh planet --env-name ClothFlatten --train-epoch 10
docker/softagent-local.sh mvp --env-name ClothFlatten
```

These are training runs, not just policy rollouts. The defaults in
`experiments/run_softgym_benchmark.py` are intentionally small for smoke tests;
increase train steps, epochs, CEM iterations, and environment variations for
paper-scale experiments.

## Direct Entrypoint

Inside the SoftAgent container or an equivalent Python environment:

```bash
python experiments/run_softgym_benchmark.py --algorithm cem --env-name ClothFlatten
```

