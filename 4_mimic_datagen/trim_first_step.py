#!/usr/bin/env python3
"""Merge generation shards into one HDF5 and drop the first N steps of every demo.

Why: Isaac Lab's ``ManagerBasedEnvCfg.num_rerenders_on_reset`` defaults to 0, so the camera image
recorded at step 0 of every Mimic-generated episode is the render of the *previous* trial's end
state (apple already on the plate, or knocked off the table), paired with the state/action of the
fresh reset. The low-dimensional data at step 0 is correct; only the image is stale. Arena's
``convert_hdf5_to_lerobot.py`` removes the *last* frame, not the first, so the stale frame would
become frame 0 of every training episode. Dropping step 0 fixes existing shards; new shards written
with ``generate_dataset_seeded.py --rerenders_on_reset N>0`` do not have the problem, and trimming
them costs one valid frame in ~500.

Also merges: demos are renumbered ``demo_0..N-1`` in input order, ``format_version`` and
``env_args`` are carried over from the first input and ``data.attrs["total"]`` is recomputed, so this
is a drop-in for ``merge_demos.py`` (run that with ``--dry_run`` first for its schema validation).

Every dataset under ``data/demo_*`` whose leading dimension equals ``num_samples`` is sliced
``[steps:]``; ``initial_state/*`` and anything else is copied verbatim. Chunking and compression are
preserved via ``create_dataset_like``.

Trimming means decompressing and recompressing every camera chunk (~1 min/GB single-threaded), so
each shard is trimmed in its own process to ``<output>.part<i>`` and the parts are then merged with
h5py's raw-chunk ``copy`` (seconds). With ``--steps 0`` the shards are raw-copied directly.

    python3 trim_first_step.py -o merged.hdf5 gen_w0.hdf5 gen_w1.hdf5 ...
    python3 trim_first_step.py -o merged.hdf5 --steps 1 --jobs 8 gen_w*.hdf5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402


def _sorted_demo_names(g: h5py.Group) -> list[str]:
    names = [k for k in g if k.startswith("demo_")]
    return sorted(names, key=lambda s: (0, int(s[5:])) if s[5:].isdigit() else (1, s))


def _copy_trimmed(src: h5py.Group, dst: h5py.Group, num_samples: int, steps: int) -> None:
    """Recursively copy ``src`` into ``dst``, slicing time-indexed datasets by ``steps``."""
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            _copy_trimmed(obj, dst.create_group(name), num_samples, steps)
            continue
        is_time_indexed = obj.ndim >= 1 and obj.shape[0] == num_samples and not name.startswith("initial_state")
        if is_time_indexed and steps > 0:
            data = obj[steps:]
            chunks = obj.chunks
            if chunks is not None:
                chunks = tuple(min(c, s) for c, s in zip(chunks, data.shape))
            d = dst.create_dataset_like(name, obj, shape=data.shape, chunks=chunks, data=data)
        else:
            d = dst.create_dataset_like(name, obj, data=obj[()])
        for k, v in obj.attrs.items():
            d.attrs[k] = v


def _trim_shard(path: str, part: str, steps: int) -> tuple[str, int, int]:
    """Worker: write a trimmed copy of one shard to ``part``. Returns (part, n_demos, n_steps)."""
    t0 = time.time()
    n_out = total = 0
    with h5py.File(path, "r") as fin, h5py.File(part, "w") as fout:
        if "format_version" in fin.attrs:
            fout.attrs["format_version"] = fin.attrs["format_version"]
        din = fin["data"]
        dout = fout.create_group("data")
        for k, v in din.attrs.items():
            if k != "total":
                dout.attrs[k] = v
        for name in _sorted_demo_names(din):
            g = din[name]
            n = int(g.attrs.get("num_samples", g["actions"].shape[0]))
            if n <= steps:
                print(f"  {os.path.basename(path)}:{name}: only {n} steps, skipped", flush=True)
                continue
            og = dout.create_group(name)
            _copy_trimmed(g, og, n, steps)
            og.attrs["num_samples"] = n - steps
            og.attrs["trimmed_leading_steps"] = steps
            total += n - steps
            n_out += 1
        dout.attrs["total"] = total
    print(f"  trimmed {os.path.basename(path)}: {n_out} demos, {total} steps ({time.time() - t0:.0f}s)", flush=True)
    return part, n_out, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="shard HDF5 files, merged in this order")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--steps", type=int, default=1, help="leading steps to drop from every demo (default 1)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel trim processes (default: one per shard, max 12)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_abs = os.path.abspath(args.output)
    for p in args.inputs:
        if os.path.abspath(p) == out_abs:
            print(f"ERROR: output equals input {p}", file=sys.stderr)
            return 2
        if not os.path.isfile(p):
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 2
    if os.path.exists(args.output) and not args.overwrite:
        print(f"ERROR: {args.output} exists; pass --overwrite", file=sys.stderr)
        return 2

    t0 = time.time()
    n_out = total = 0
    sources = list(args.inputs)
    parts: list[str] = []
    tmp = args.output + ".tmp"
    try:
        if args.steps > 0:
            jobs = args.jobs or min(len(sources), 12)
            print(f"trimming {len(sources)} shards with {jobs} processes")
            parts = [f"{args.output}.part{i}" for i in range(len(sources))]
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                list(ex.map(_trim_shard, sources, parts, [args.steps] * len(sources)))
            sources = parts

        print(f"merging {len(sources)} files -> {args.output}")
        with h5py.File(tmp, "w") as fout:
            dout = fout.create_group("data")
            for i, path in enumerate(sources):
                with h5py.File(path, "r") as fin:
                    din = fin["data"]
                    if i == 0:
                        if "format_version" in fin.attrs:
                            fout.attrs["format_version"] = fin.attrs["format_version"]
                        for k, v in din.attrs.items():
                            if k != "total":
                                dout.attrs[k] = v
                    for name in _sorted_demo_names(din):
                        g = din[name]
                        fin.copy(g, dout, name=f"demo_{n_out}")  # raw chunk copy, no recompression
                        total += int(g.attrs.get("num_samples", g["actions"].shape[0]))
                        n_out += 1
            dout.attrs["total"] = total
        os.replace(tmp, args.output)
    finally:
        for p in parts + [tmp]:
            if os.path.exists(p):
                os.unlink(p)

    env_args = {}
    with h5py.File(args.output, "r") as f:
        if "env_args" in f["data"].attrs:
            env_args = json.loads(f["data"].attrs["env_args"])
    print(f"\nwrote {n_out} demos, {total} steps (dropped {args.steps} leading step(s) each) -> {args.output} "
          f"({os.path.getsize(args.output) / 1e9:.2f} GB, {time.time() - t0:.0f}s) env={env_args.get('env_name')}")
    return 0 if n_out else 1


if __name__ == "__main__":
    sys.exit(main())
