# Attribution

This project builds on and references several external research codebases,
simulation libraries, and baseline implementations. This file summarizes those
references for project documentation

## Primary Project References

- LeWorldModel / LeWM: https://github.com/lucas-maes/le-wm
  - Used as the main architectural and training reference for the
    action-conditioned JEPA world model, which we have adapted to 
    predict action conditioned dynamics in the a deformbale object 
    simulation environment

- stable-worldmodel: https://github.com/galilai-group/stable-worldmodel
  - Referenced for world-model evaluation, MPC policy structure, CEM solver
    interfaces, and related training/evaluation patterns.

## Simulation And Benchmark Code

- SoftGym: https://github.com/Xingyu-Lin/softgym
  - under `simulation/softgym/` and adapted for this project's
    deformable-object data collection and evaluation workflows.
  - License file present at `simulation/softgym/LICENSE.txt`.

- SoftAgent: https://github.com/Xingyu-Lin/softagent
  - under `simulation/softagent/` and used for SoftGym benchmark
    algorithms and CEM expert collection.
  - Local usage notes are in `simulation/softagent/LOCAL_USAGE.md`.

- NVIDIA FleX: https://github.com/NVIDIAGameWorks/FleX
  - The underlying particle simulator used by SoftGym/PyFlex.
  - Local license file present at `simulation/PyFlex/LICENSE.txt`.

- PyFleX: https://github.com/YunzhuLi/PyFleX
  - Python bindings/interface layer used by SoftGym, which 
    was used for docker containers in this project


## Other Referenced Repositories And Tools

For package-level dependencies used in this project, see the relevant
`requirements.txt` files in the repository, especially `training/requirements.txt`
and the Docker-specific requirements files under `simulation/docker/`. Upstream directories may include their own requirements files.

## Compute Resources

Experiments and development used a mix of personal compute and externally provided compute resources. Compute was provisioned or supported by AWS, Google Cloud, NVIDIA, and Stanford University's CS231N.

## AI Use

AI assistance tools, including OpenAI's ChatGPT/Codex and Anthropic's Claude Code, were used during development for tasks such as code drafting, debugging, merge-conflict resolution, experiment script iteration, documentation drafting, and summarizing results. Human project members directed the work, reviewed generated changes, ran experiments, and made the final decisions about architecture choices, training runs, model comparisons, and what to include in the repository. Where AI chat logs and outputs were preserved, they are stored under the `ai_docs` folder. Due to multiple compactions and agent sessions a full log is not stored, but at a high level, AI assistance was used to help draft and revise shell scripts for launching multi-GPU data collection and single-GPU training runs, assist with PyTorch syntax and formatting, quickly generate plotting and charting utilities, prototype implementation ideas, and iterate on experiment orchestration code.
