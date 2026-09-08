"""Log the pelvis pose and the runtime physics parameters, for any embodiment.

Both are needed to compare two robots that should behave identically and do not. The pelvis trace
answers "is the base moving differently"; the physics dump answers "is the plant actually the
same", which is the question that mattered here: every property compared offline in the URDF and
the USD matched between the two robots, yet the pelvis settled differently from the first step.
The runtime values are what the solver integrates, so that is where to look.

Also logs the whole-body command slots the policy emits alongside the 43 joint targets, since the
base-height and torso-orientation commands are what would make the controller crouch or lean and
are therefore the first thing to rule out.

Output prefixes, all greppable across runs:

    PHYSDOF     per joint: stiffness, damping, armature, friction, max velocity, max force, limits
    PHYSLINK    per body: mass and centre of mass
    PHYSSOLVER  solver iteration counts
    PHYSDEFQ    default joint positions, the controller's posture reference
    PHYSPOS0    pelvis and ankle positions at reset
    ROOTEP      one summary line per episode
    ROOT_TRACE  pelvis x and z through the episode

Copy into the IsaacLab-Arena checkout root and run with /isaac-sim/python.sh; see README.md.
"""

import numpy as np
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)
from isaaclab_arena_gr00t.utils.groot_path import ensure_groot_deps_in_path


def _np(x):
    """Isaac Lab exposes some of this data as warp arrays and some as torch tensors."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    try:
        import warp as wp

        return wp.to_torch(x).detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


def root_pos(env) -> np.ndarray:
    return _np(env.unwrapped.scene["robot"].data.root_link_pos_w)[0]


def dump_runtime_physics(env) -> None:
    """Print what PhysX actually loaded for the robot."""
    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    names = list(robot.joint_names)

    getters = {
        "stiff": "get_dof_stiffnesses",
        "damp": "get_dof_dampings",
        "arm": "get_dof_armatures",
        "fric": "get_dof_friction_coefficients",
        "maxvel": "get_dof_max_velocities",
        "maxF": "get_dof_max_forces",
        "lim": "get_dof_limits",
    }
    cols = {}
    for key, getter in getters.items():
        try:
            cols[key] = _np(getattr(view, getter)())[0]
        except Exception as exc:  # noqa: BLE001
            print(f"PHYSWARN {getter}: {exc}", flush=True)

    for i, name in enumerate(names):
        parts = [f"PHYSDOF {name:32s}"]
        for key, values in cols.items():
            if key == "lim":
                parts.append(f"lim=({values[i][0]:+.4f},{values[i][1]:+.4f})")
            else:
                parts.append(f"{key}={float(values[i]):.5g}")
        print(" ".join(parts), flush=True)

    try:
        masses = _np(view.get_masses())[0]
        coms = _np(view.get_coms())[0]
        for name, m, c in zip(robot.body_names, masses, coms):
            print(f"PHYSLINK {name:32s} m={m:.4f} "
                  f"com=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})", flush=True)
        print(f"PHYSTOTAL mass={masses.sum():.4f} bodies={len(masses)} dofs={len(names)}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"PHYSWARN masses: {exc}", flush=True)

    try:
        print(f"PHYSSOLVER pos_iters={_np(view.get_solver_position_iteration_counts())[0]} "
              f"vel_iters={_np(view.get_solver_velocity_iteration_counts())[0]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"PHYSWARN solver: {exc}", flush=True)

    for name, value in zip(names, _np(robot.data.default_joint_pos)[0]):
        print(f"PHYSDEFQ {name:32s} {value:+.4f}", flush=True)

    try:
        positions = _np(robot.data.body_link_pos_w)[0]
        for name, p in zip(robot.body_names, positions):
            if "ankle_roll" in name or name == "pelvis":
                print(f"PHYSPOS0 {name:24s} "
                      f"({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"PHYSWARN pos0: {exc}", flush=True)


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument("--episodes", type=int, default=8)
    args_parser.add_argument("--every", type=int, default=10, help="sample the pose every N steps")
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()
        policy_cls = get_policy_cls(args_cli.policy_type)
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = policy_cls.add_args_to_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        env, _ = arena_builder.make_registered_and_return_cfg()
        policy = policy_cls.from_args(args_cli)

        obs, _ = env.reset()
        policy.reset()
        dump_runtime_physics(env)
        policy.set_task_description(
            args_cli.language_instruction
            or env.unwrapped.cfg.isaaclab_arena_env.task.get_task_description()
        )

        step, done = 0, 0
        trace: list[np.ndarray] = []
        cmd_trace: list[np.ndarray] = []

        pbar = tqdm.tqdm(total=args_cli.episodes, desc="Episodes", unit="ep")
        while done < args_cli.episodes:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)
                step += 1
                if step % args_cli.every == 1:
                    trace.append(root_pos(env))
                    cmd_trace.append(actions[0, 43:].detach().cpu().numpy())

                if terminated.any() or truncated.any():
                    done += 1
                    t = np.array(trace)
                    c = np.array(cmd_trace)
                    success = bool(
                        env.unwrapped.termination_manager.get_term("success")[0].item()
                    )
                    # One self-contained line per episode, prefixed so it survives tqdm's output.
                    print(
                        f"\nROOTEP {done:02d} ok={int(success)} "
                        f"x0={t[0, 0]:+.3f} x_end={t[-1, 0]:+.3f} x_min={t[:, 0].min():+.3f} "
                        f"y_end={t[-1, 1]:+.3f} "
                        f"z0={t[0, 2]:+.3f} z_end={t[-1, 2]:+.3f} z_min={t[:, 2].min():+.3f} "
                        f"cmd_mean={np.array2string(c.mean(0), precision=3, separator=',')} "
                        f"cmd_min={np.array2string(c.min(0), precision=3, separator=',')} "
                        f"cmd_max={np.array2string(c.max(0), precision=3, separator=',')}",
                        flush=True,
                    )
                    print("ROOT_TRACE_X " + " ".join(f"{v:+.3f}" for v in t[:, 0]), flush=True)
                    print("ROOT_TRACE_Z " + " ".join(f"{v:+.3f}" for v in t[:, 2]), flush=True)
                    trace, cmd_trace, step = [], [], 0
                    pbar.update(1)
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
        pbar.close()

        if getattr(env.unwrapped.cfg, "metrics", None) is not None:
            from isaaclab_arena.metrics.metrics import compute_metrics

            print(f"FINAL METRICS: {metrics_to_plain_python_types(compute_metrics(env.unwrapped))}")
        if policy.is_remote:
            policy.shutdown_remote(kill_server=False)
        env.close()


if __name__ == "__main__":
    ensure_groot_deps_in_path()
    main()
