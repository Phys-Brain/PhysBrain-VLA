# PhysBrain-VLA

PhysBrain 1.0 — Physical Intelligence for Embodied General AI

[Open Source Plan](#open-source-plan) • [Overview](#overview) • [Key Technologies](#key-technologies) • [Model Zoo](#model-zoo) • [Getting Started](#getting-started)

---

## Overview

**PhysBrain 1.0** is a physical intelligence system built around the paradigm shift from *action imitation* to *physical commonsense acquisition* in embodied AI. At its core is the world's first scalable **physical intelligence data engine** capable of converting large-scale human egocentric video into high-quality multimodal training data — eliminating the dependency on expensive closed-loop robot demonstrations.

The data engine processes over **3,000 hours** of human video with fine-grained annotations across spatial relationships, action feasibility, and multi-step logical reasoning in real 3D environments. This corpus moves beyond simple action replication to extract the underlying physical laws and commonsense logic embedded in everyday human activity. When injected into a multimodal large model, it successfully elicits **human-like physical intelligence**, enabling the model to *understand physics* rather than merely imitate motions.

The resulting PhysBrain base model achieves **state-of-the-art (SOTA) performance** across multiple authoritative benchmarks in spatial intelligence and embodied interaction.

Built upon this foundation, this repository provides the **Vision-Language-Action (VLA)** model for robot control — the bridge from physical intelligence to real-world robotic applications.

## Key Technologies

### PhysBrain Data Engine

A scalable pipeline that transforms raw human egocentric video into structured, multimodal embodied training data — annotated with spatial structure, motion feasibility, and causal reasoning chains. This zero-cost approach to **physical commonsense injection** removes the bottleneck of robot-collected data and enables training at unprecedented scale.

### TwinBrainVLA — Dual-Brain Fusion Architecture

A novel architecture that addresses the industry-wide challenge of **catastrophic forgetting** during embodied fine-tuning. By maintaining a parallel general-purpose brain alongside a task-specific embodied brain, TwinBrainVLA achieves **"generalist-specialist fusion"** — retaining broad semantic understanding while efficiently acquiring domain-specific embodied skills.

### LangForce — Physics-Grounded Training Strategy

A principled training methodology that breaks the **visual shortcut dilemma** in VLA learning through a Bayesian statistical lens. LangForce fundamentally shifts the training objective from behavioral cloning to **physical commonsense acquisition**, enabling the model to reason about and adapt to complex physical scenarios rather than memorizing action sequences.

Together, these three technologies form the PhysBrain 1.0 system: **PhysBrain** (base model) × **TwinBrainVLA** (architecture) × **LangForce** (training strategy) — achieving SOTA across multiple embodied intelligence benchmarks with exceptional data efficiency.

## Open Source Plan

We are progressively open-sourcing PhysBrain 1.0 components. The current release status is as follows:

| Component                                              | Status       |
| ------------------------------------------------------ | ------------ |
| PhysBrain 1.0 VLA (RoboCasa Fine-Tuned)                | ✅ Available  |
| PhysBrain 1.0 VLA (LIBERO Fine-Tuned)                  | Coming Soon  |
| PhysBrain 1.0 VLA (SIMPLER WidowX Robot Fine-Tuned)   | Coming Soon  |
| PhysBrain 1.0 VLA (SIMPLER Google Robot Fine-Tuned)    | Coming Soon  |
| Inference Code                                         | Coming Soon  |
| Deployment Code (RoboCasa Fine-Tuned)                 | ✅ Available  |

## Model Zoo

### PhysBrain 1.0 VLA (RoboCasa Fine-Tuned)

Fine-tuned on the [RoboCasa](https://robocasa.ai/) benchmark suite for household manipulation tasks.

| Model | Download |
| --- | --- |
| PhysBrain-1.0-VLA-RoboCasa | [🤗 Hugging Face](https://huggingface.co/collections/Phys-Brain/physbrain-10-vla) |

> Deployment code for RoboCasa fine-tuned model is now available in this repository. Inference code will be released alongside upcoming model checkpoints. Stay tuned.

## Getting Started

> Code is coming soon. This section will be updated with installation instructions, usage examples, and API documentation upon release.

## Citation

If you find PhysBrain useful in your research, please consider citing our work:

```bibtex
TODO
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
