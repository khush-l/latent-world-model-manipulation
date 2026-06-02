# Oracle-CEM expert collection — runbook (multi-GPU / many-CPU box)

Collect SoftAgent oracle-CEM expert RopeFlatten demonstrations (ground-truth sim
as the planner's model → near-optimal, diverse flattening incl. the grasp), record
them into the v3 NPZ → HDF5 schema, and merge into a training dataset.

**Validated on the L4 dev box:** CEM flattens to 0.998–0.999; the recorder
(`replay_cem_traj.py`) reproduces it (0.998) into the correct schema; the full glue
(CEM → replay → `npz_to_hdf5` → `merge_h5`) works end-to-end. What changes on the big
box is **scale + GPU spread** — see below.

CEM is **CPU-bound** (each episode = an 8-process rollout Pool stepping FleX serially;
the GPU per env is light). Throughput ≈ total busy FleX worker processes ≈ vCPUs.
A 4×L40 / 126-vCPU box → ~12–15 concurrent shards → **2,000 episodes in ~roughly a
day** (slim budget). The L40s aren't the limiter — cores are — but 4 GPUs let us
**spread FleX envs so no single GPU chokes** (a 1-GPU L4 already contends at ~16 envs).

---

## 0. Prereqs (one-time, on the box)

```bash
cd <repo>/simulation
docker/softgym-local.sh build        # softgym image
docker/softagent-local.sh build      # softagent image (CEM)
docker/softagent-local.sh compile    # builds PyFlex .so (auto-runs on first use too)
nproc; free -g; nvidia-smi -L        # confirm ~126 vCPU, RAM, 4 GPUs
```

## 1. Sanity: one CEM episode flattens (~6–9 min)

```bash
docker/softagent-local.sh cem --env-name RopeFlatten --test-episodes 1 \
  --max-iters 3 --timestep-per-decision 900 --save-video True \
  --exp-name sanity --log-dir data/sanity
# expect log: info_final_normalized_performance ~ 0.95–0.999
# gif: simulation/softagent/data/sanity/RopeFlatten.gif
```

## 2. Parallelism sweep — find MAX_PAR (do this BEFORE the full run)

The one number to calibrate. For each level, launch that many **GPU-pinned**
1-episode shards, time the wave, and watch RAM + **per-GPU** util:

```bash
# in another terminal, keep these open while sweeping:
watch -n2 nvidia-smi          # want ALL 4 GPUs busy, none pegged at 100% queue
watch -n2 free -g             # watch 'used' — must stay well under total RAM

# sweep (each takes ~6–9 min; pick a level then Ctrl-C the rest):
for N in 4 8 12 15; do
  echo "=== MAX_PAR=$N ==="; t0=$(date +%s)
  SMOKE=0 TOTAL_EPISODES=$N SHARDS=$N MAX_PAR=$N NUM_GPUS=4 \
    MAX_ITERS=3 TPD=900 OUT_H5=/tmp/sweep_$N.h5 \
    bash simulation/collect_cem_parallel.sh >/tmp/sweep_$N.log 2>&1
  echo "MAX_PAR=$N : $N eps in $(( $(date +%s)-t0 ))s"
done
```
Pick the **highest MAX_PAR where episodes/sec still scales** AND RAM stays safe AND
no single GPU is saturated. Expected sweet spot on this box: **MAX_PAR ≈ 12–15**
(126/8 ≈ 15 cores cap). If RAM climbs near total or a GPU pegs, back off.

**Rules of thumb / what to look for:**
- **RAM**: ~1 GB per FleX env; MAX_PAR×8 envs. 12×8=96 envs ≈ ~100 GB — confirm headroom.
- **GPUs**: all 4 should share load (~3 shards/GPU at MAX_PAR=12). If only GPU 0 is
  busy → the `CUDA_VISIBLE_DEVICES` pinning isn't taking (see Troubleshooting).
- **Scaling**: if going 8→12 doesn't raise eps/sec, you're GPU/RAM-bound — stay at 8.

## 3. Smoke the full pipeline (~15 min)

```bash
SMOKE=1 MAX_ITERS=3 TPD=900 bash simulation/collect_cem_parallel.sh
# validates CEM -> replay -> npz_to_hdf5 -> merge end-to-end (4 eps).
# expect: 'DONE -> .../rope_cem_dataset.h5' and a row/episode count printout.
```

## 4. Launch the full collection in tmux

```bash
tmux new -s cem "TOTAL_EPISODES=2000 SHARDS=200 MAX_PAR=12 NUM_GPUS=4 \
  MAX_ITERS=3 TPD=900 OUT_H5=simulation/data/rope/rope_cem_2k.h5 \
  bash simulation/collect_cem_parallel.sh 2>&1 | tee /tmp/cem_run.log"
# detach: Ctrl-b d   |   reattach: tmux attach -t cem
```
- **SHARDS=200 → 10 eps/shard**: a `shard.h5` is written every ~10 episodes, so the
  run is **interrupt-safe** — completed shards survive a crash/stop.
- Distinct `--seed` per shard → distinct ropes (no duplicate data).

## 5. Monitor — what to look for

```bash
tmux attach -t cem                                   # live
ls simulation/data/rope_cem/shard*.h5 | wc -l        # completed shards (×10 = eps done)
tail -f simulation/data/rope_cem/logs/shard*.log     # per-shard CEM/replay progress
grep final_perf simulation/data/rope_cem/logs/*.log  # quality per episode (~0.95+)
nvidia-smi          # all 4 GPUs busy, balanced
free -g             # 'used' stable, not creeping toward total (else OOM risk -> lower MAX_PAR)
```
**Healthy run:** every shard log shows `episode i, step ...` then `ep N -> ...npz
(final_perf ~0.97)` then `[shard i] DONE`; new `shard*.h5` appear steadily; RAM flat;
4 GPUs balanced. **Red flags:** RAM creeping up (OOM coming → lower MAX_PAR), only GPU
0 busy (pinning broken), `final_perf` < ~0.8 often (CEM budget too slim → raise TPD).

## 6. After collection — build the training set

```bash
# CEM expert + a slice of existing random (off-policy coverage). Adjust counts.
python simulation/utils/merge_h5.py \
  --output simulation/data/rope/rope_lewm_dataset.h5 \
  --input simulation/data/rope/rope_cem_2k.h5:cem \
  --input simulation/data/rope/khush_random_rope_5k75.h5:random   # (subset if desired)
```
Target dataset: **~1,000–2,000 CEM + ~1,000 random**. Then train (gray, proprio),
**~40–60k steps** (NOT 250k — ~150k–300k frames overfits past that); checkpoint every
5–10k, eval with the LeWM-matched `eval/mpc_runner.py`. See `RESULTS.md`.

---

## Troubleshooting

- **Only GPU 0 busy / others idle.** The per-shard `CUDA_VISIBLE_DEVICES` pin isn't
  reaching the container. Confirm the docker wrappers forward it (they do, after the
  `-e CUDA_VISIBLE_DEVICES` patch) and that you didn't override `NUM_GPUS`. Test:
  `CUDA_VISIBLE_DEVICES=2 docker/softgym-local.sh run "nvidia-smi -L"` should still list
  all GPUs but the process should land on GPU 2.
- **OOM / RAM climbing.** Lower `MAX_PAR` (fewer concurrent FleX envs). Each env ~1 GB.
- **A shard fails.** Other shards continue (failures are isolated). Its log is at
  `simulation/data/rope_cem/logs/shardN.log`; re-run just that shard by hand if needed.
- **`cem_traj.pkl` missing for a shard.** The CEM container died before finishing its
  10 episodes (run_cem saves the pkl only at the end of its `--test-episodes`). Smaller
  `EP_PER_SHARD` = less lost on a crash (already 10).
- **Cross-container paths:** CEM writes under `simulation/softagent/data/...` (softagent
  workdir), replay reads it as `softagent/data/...` (softgym workdir). The orchestrator
  handles this; don't change `CEM_BASE`/`NPZ_BASE` without re-checking both wrappers.
- **Slow / per-GPU choke.** If a GPU pegs with many envs, lower `MAX_PAR` so
  `MAX_PAR/NUM_GPUS` (shards/GPU) drops — fewer FleX envs per GPU.
