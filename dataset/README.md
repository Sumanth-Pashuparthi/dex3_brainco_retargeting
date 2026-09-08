# Dataset

Published Hugging Face dataset for this project:

**https://huggingface.co/datasets/pashuparthis/mimic_apple_pick_and_place**

200 Isaac Lab Mimic episodes of Unitree G1 + BrainCo Revo2 apple pick-and-place (LeRobot v2 /
GR00T N1.7 layout). The heavy files are not in git — this folder is the in-repo pointer.

| | |
|---|---|
| Episodes | 200 |
| Frames | 51 746 at 50 Hz |
| Size on Hub | ~129 MB LeRobot tree (raw `gen_w*.hdf5` shards are larger) |
| Camera | head `ego_view`, 640×480 |
| Pipeline | [`4_mimic_datagen/`](../4_mimic_datagen/) |

## Get the data here

```bash
cd dataset
./fetch.sh
# -> ./mimic_apple_pick_and_place/   (LeRobot v2: data/, meta/, videos/)
```

Or manually:

```bash
hf download pashuparthis/mimic_apple_pick_and_place \
  --repo-type dataset \
  --local-dir dataset/mimic_apple_pick_and_place
```

If you already have a local copy elsewhere, link it instead of re-downloading:

```bash
ln -sfn /path/to/mimic_apple_pick_and_place dataset/mimic_apple_pick_and_place
```

`mimic_apple_pick_and_place/` is gitignored. Only this README and `fetch.sh` are tracked.
