#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-${1:-/home/ubuntu/openpi}}
CONFIG_PY="$OPENPI_ROOT/src/openpi/training/config.py"
POLICY_DST="$OPENPI_ROOT/src/openpi/policies/softgym_policy.py"

if [[ ! -f "$CONFIG_PY" ]]; then
  echo "Missing OpenPI config.py at $CONFIG_PY" >&2
  echo "Run baselines/pi0/bootstrap_openpi.sh $OPENPI_ROOT first." >&2
  exit 1
fi

cp "$REPO_ROOT/baselines/pi0/openpi_softgym_policy.py" "$POLICY_DST"

if ! grep -q "softgym_policy" "$CONFIG_PY"; then
  sed -i '/import openpi.policies.libero_policy as libero_policy/a import openpi.policies.softgym_policy as softgym_policy' "$CONFIG_PY"
fi
if ! grep -q "^import os$" "$CONFIG_PY"; then
  sed -i '/import logging/a import os' "$CONFIG_PY"
fi

if ! grep -q "class LeRobotSoftGymDataConfig" "$CONFIG_PY"; then
  python3 - "$CONFIG_PY" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
marker = "\n\n@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig"
block = r'''

@dataclasses.dataclass(frozen=True)
class LeRobotSoftGymDataConfig(DataConfigFactory):
    """SoftGym Rope/Cloth LeRobot data for OpenPI fine-tuning."""

    default_prompt: str | None = "straighten the rope"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/front": "observation.images.front",
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[softgym_policy.SoftGymInputs(model_type=model_config.model_type)],
            outputs=[softgym_policy.SoftGymOutputs(action_dim=8)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
'''
if marker not in text:
    raise SystemExit("Could not find insertion marker for LeRobotSoftGymDataConfig")
path.write_text(text.replace(marker, block + marker))
PY
fi

if ! grep -q 'name="pi05_softgym_rope_lora"' "$CONFIG_PY"; then
  python3 - "$CONFIG_PY" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
marker = "    #\n    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.\n"
block = r'''    TrainConfig(
        name="pi05_softgym_rope_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotSoftGymDataConfig(
            repo_id=os.environ.get("OPENPI_DATASET_REPO_ID", "khush/softgym-rope-openpi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=int(os.environ.get("OPENPI_NUM_TRAIN_STEPS", "20_000")),
        batch_size=int(os.environ.get("OPENPI_BATCH_SIZE", "4")),
        num_workers=int(os.environ.get("OPENPI_NUM_WORKERS", "4")),
        save_interval=int(os.environ.get("OPENPI_SAVE_INTERVAL", "1000")),
        keep_period=int(os.environ.get("OPENPI_KEEP_PERIOD", "5000")),
    ),
'''
if marker not in text:
    raise SystemExit("Could not find insertion marker for pi05_softgym_rope_lora TrainConfig")
path.write_text(text.replace(marker, block + marker))
PY
fi

echo "Installed SoftGym OpenPI policy/config into $OPENPI_ROOT"
echo "Config name: pi05_softgym_rope_lora"

