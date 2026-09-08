# Transferring a dexterous manipulation policy from Unitree Dex3-1 to BrainCo Revo2 Touch hands

A frozen GR00T N1.7 policy, trained on a Unitree G1 29-DoF with **7-DoF Dex3-1 hands**, driving the
same robot with **6-DoF BrainCo Revo2 Touch hands** — without retraining, without touching the
policy, and without changing its action space.

|  | Dex3-1 (source) | Revo2 Touch (target) |
|---|---|---|
| Actuated DoF per hand | 7 | 6 |
| Fingers | 3 | 5 |
| Robot DoF | 43 | 41 |
| Task success | **0.65** (13/20) | **0.06** (6/100, all frame-verified) |

The task, scene, checkpoint, success criterion, episode length and evaluation script are identical
between the two columns. The only difference is one command-line argument. The drop is the expected
and useful result: this configuration is a measurement of the embodiment gap, which is what tells
you where fine-tuning has to be aimed.

[`g1_brainco_gr00t_inference.mp4`](g1_brainco_gr00t_inference.mp4) is one of the six verified Revo2
successes (episode 57 of the 100-episode evaluation, head camera, 6 s): the frozen Dex3 policy,
through the retargeting layer, grasping the apple with the Revo2 hand on the first attempt, lifting
it, carrying it and placing it on the plate. The other five, and the false-success counter-example,
are in [`3_g1_brainco_inference/media/`](3_g1_brainco_inference/media/).

## Repository layout

Four steps, in the order they were done. Each has its own README with commands and expected output.

| Step | Contents |
|---|---|
| [`1_baseline/`](1_baseline/) | Run the existing apple pick-and-place unmodified. Establishes the reference number. |
| [`2_retargeting/`](2_retargeting/) | Build the Revo2 robot asset, and the layer that maps Dex3 commands onto it. |
| [`3_g1_brainco_inference/`](3_g1_brainco_inference/) | Run the policy on the retargeted robot; record successes and instrument the physics comparison. |
| [`4_mimic_datagen/`](4_mimic_datagen/) | Harvest the policy's successes and multiply them with Isaac Lab Mimic into a fine-tuning dataset. |
| [`dataset/`](dataset/) | Pointer to the HF dataset ([pashuparthis/mimic_apple_pick_and_place](https://huggingface.co/datasets/pashuparthis/mimic_apple_pick_and_place)); `./fetch.sh` downloads it (or use the local symlink). |
| [`docs/assignment1_report.pdf`](docs/assignment1_report.pdf) | The write-up (5 pages): platform choice, the correspondence, asset changes, results and sim-to-real notes. |
| [`assignment2_report.pdf`](assignment2_report.pdf) | Assignment 2 (2 pages): the process for teaching a new right-to-left handover skill on the same G1 + Revo2 stack. |
| [`g1_brainco_gr00t_inference.mp4`](g1_brainco_gr00t_inference.mp4) | Headline clip: the retargeted policy completing the task on the Revo2 hand (same as `3_g1_brainco_inference/media/success_ep057.mp4`). |

The generated dataset is not committed (`.gitignore` covers `*.hdf5`, `*.parquet`, `lerobot/`). The
raw Mimic output is published on Hugging Face as the six per-worker HDF5 shards, exactly as
`generate_dataset.py` wrote them:

**[pashuparthis/mimic_apple_pick_and_place](https://huggingface.co/datasets/pashuparthis/mimic_apple_pick_and_place)**
— `gen_w0.hdf5` … `gen_w5.hdf5`, 200 successful episodes (33–34 per shard), 13.4 GB, Arena
`record_demos.py` schema: 23-D Pink actions, states, observations and the 640×480 head camera at 50 Hz.

```bash
hf download pashuparthis/mimic_apple_pick_and_place --repo-type dataset --local-dir ./mimic_apple_pick_and_place
cd 4_mimic_datagen && ./run_merge.sh && ./run_convert.sh   # -> one HDF5, then GR00T-LeRobot v2
```

Steps 1 to 3 are complete and their numbers are measured. Step 4 is complete through data
generation: 9 source demos annotated, 200 demos generated with Mimic (61% generation success),
converted and validated with GR00T's loader. The fine-tuning run itself is not done.

## The write-up

[docs/assignment1_report.pdf](docs/assignment1_report.pdf) is the full account in five pages: why
this platform and task, what the two hands differ by, how the correspondence was derived, what
changed in the asset, the results with the ablation and the pelvis investigation, the
data-generation pipeline, and what deployment on hardware would need.

[assignment2_report.pdf](assignment2_report.pdf) is the two-page process write-up for Assignment 2:
teaching the same stack a new skill on the Revo2 hands (pick with the right hand, hand over to the
left, put down, with the release triggered by the receiving hand's contact rather than a timer).

## Setup

Requires an NVIDIA GPU with ~32 GB VRAM (an RTX 5090 was used), Docker with the NVIDIA container
runtime, and about 80 GB of disk.

Installation is scripted and lives with step 1, which is self-contained:

```bash
docker login nvcr.io          # username is the literal string $oauthtoken, password is your NGC API key
cd 1_baseline
./setup.sh
```

That pulls the Isaac Sim image, clones and builds IsaacLab-Arena at `release/0.2.1`, clones
Isaac-GR00T at the pinned commit `4b1dca9`, downloads the checkpoint, and wires the policy config to
it. Roughly an hour, almost all of it downloading. Every step is skipped if already done, so
re-running after a failure resumes. Everything lands under `~/g1_baseline`; override with `WORK_DIR`.

See [`1_baseline/README.md`](1_baseline/README.md) for the full prerequisites and what each of the
six install steps does.

## Run everything

```bash
# The inference server runs on the host and is shared by every step that runs a policy.
(cd 1_baseline && ./run_policy_server.sh)      # leave running

# 1. baseline: expect success_rate 0.65
(cd 1_baseline && ./run_eval.sh 100)

# 2. build the Revo2 asset and install the retargeting layer
(cd 2_retargeting && ./build_asset.sh && ./install.sh && python3 test_retargeting.py)

# 3. run the policy on the retargeted robot: expect success_rate ~0.06
(cd 3_g1_brainco_inference && ./run_eval.sh 100)

# 4. build demonstration data for fine-tuning
(cd 4_mimic_datagen && ./setup.sh && ./prepare_source.sh)
```

Each step folder is self-contained: its own `env.sh`, container runner and media, with no references
outside itself. Installation lives in steps 1 and 2, which share the same `WORK_DIR` defaults, so
whichever you install first makes the other's setup a no-op. Everything lands under `~/g1_baseline`.

## What the retargeting layer actually does

Seven independently actuated joints per hand collapse to six, two of which have no source signal at
all, and each Revo2 finger has to represent a two-joint Dex3 flexion chain with a single actuator.
Angle conventions differ too, so the correspondence is defined on **closure fraction** rather than
angle, and calibrated on **fingertip aperture** rather than closure fraction — the Dex3 opens to
193 mm and the Revo2 only to 141 mm, so matching fractions closes the hand on empty air.

Four things then have to be right at once:

- **Aperture calibration.** Match openings, not fractions, on the running minimum so the map stays
  monotone. Without it the hand closes on empty air in every episode.
- **Thumb opposition.** The Revo2's metacarpal sweeps across the palm rather than rolling like the
  Dex3's thumb. A geometric schedule swings it into opposition *ahead* of the fingers. Pinch aperture
  at 75% closure: 36.9 mm → 5.5 mm, without which no grasp is geometrically possible.
- **Invertibility.** Strict aperture matching leaves a dead zone where commanded closure produces no
  motion, so the policy's own command never reappears in its next observation. Visible in logs as
  the policy opening and re-closing mid-grasp.
- **Interface preservation.** The action space stays 50-D in Dex3 coordinates. The 41-DoF
  articulation is reached only inside the action term, so the policy, its client, the modality
  config and the whole-body controller are all untouched.

Full derivation in [docs/assignment1_report.pdf](docs/assignment1_report.pdf) (section "The correspondence").

## The finding that mattered most was not in the hand

Debug traces showed the pelvis creeping 5.7 cm backwards every episode and settling 2.4 cm lower,
where the baseline settles in 0.4 s around a centred +0.5 cm mean. That turns the top-down grasp
approach the policy learned into a side sweep. Every authored property matched between the two
robots: per-DoF gains, armature, friction, limits, every link mass. And the pelvis still settled
differently from the first step.

The scene has an invisible collision slab under the table and the robot spawns overlapping it. The
stock G1 has a `pelvis_contour_link` collider and braces against it, which is why the baseline's 65%
is so repeatable. The supplied BrainCo URDF omits that link, so its pelvis had no collider at all
and what looked like drift was just the free-standing controller. Restoring it took true
pick-and-place from 1/100 to 6/100.

Six other hypotheses were tested and rejected along the way, including two that seemed much more
likely; they are recorded in [docs/assignment1_report.pdf](docs/assignment1_report.pdf) (section "The pelvis").

## Reading the numbers

The task's success term is "apple within the plate region", which also counts the apple being
*pushed* there. In one 100-episode run only 1 of 3 counted successes was a real grasp, so every
number in this repository was checked frame by frame at 4 fps.

Two more caveats worth knowing before quoting anything:

- **Rates below 100 episodes are noise.** One configuration scored 1/10 and then 0/20 on repeat.
- **The apple spawn is deterministic** (`APPLE_SPAWN_XY_RANGE_M = 0.0`). The task measures
  repeatability under physics and policy stochasticity, not spatial generalisation.

## Components and licences

| Component | Version | Licence |
|---|---|---|
| Isaac Sim | 6.0.0-dev2 | NVIDIA Omniverse EULA |
| IsaacLab-Arena | `release/0.2.1` | Apache 2.0 |
| Isaac-GR00T | `4b1dca9` | Apache 2.0 (code) |
| GR00T N1.7 checkpoint | `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` | NVIDIA Open Model Licence |
| Unitree G1 / Dex3-1 assets | as shipped with Arena | Unitree terms |
| BrainCo Revo2 URDF and meshes | as supplied | BrainCo terms |

Code in this repository is the retargeting layer, the asset patches, the analysis and the
instrumentation. Everything else is upstream and unmodified except where
[docs/assignment1_report.pdf](docs/assignment1_report.pdf) (section "The asset") says otherwise.
