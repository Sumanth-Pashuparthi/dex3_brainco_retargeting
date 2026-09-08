"""Convert the G1 29-DoF + BrainCo Revo2 URDF into a USD that Isaac Lab can spawn.

The baseline robot ships as g1_29dof_with_hand_rev_1_0.usd (Unitree Dex3-1, 7 actuated joints per
hand). This produces the target robot with the Revo2 Touch hand (6 actuated joints per hand, distal
phalanges rigid in this URDF), preserving link inertials, collision geometry and joint limits.

merge_fixed_joints stays on: the Revo2's distal and tip joints are authored as `fixed` here, so
merging folds them into their parent links rather than creating zero-DoF articulation entries.
self_collision stays off to match the baseline G1 articulation settings.

The converter output needs three corrections before it behaves like the baseline asset; run
patch_converted_usd.py afterwards.

Run inside the Isaac Sim container with /isaac-sim/python.sh. See README.md.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True, help="path to g1_29dof_with_brainco_hand.urdf")
parser.add_argument("--out", required=True, help="output .usd/.usda path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=args.urdf,
        usd_dir=out_dir,
        usd_file_name=os.path.basename(args.out),
        fix_base=False,  # floating-base humanoid; the whole-body controller balances it
        merge_fixed_joints=True,
        self_collision=False,
        collision_type="Convex Hull",
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            # Placeholder gains. Isaac Lab's ArticulationCfg actuator groups override these at
            # spawn time, so the authoritative stiffness and damping come from the robot config.
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=5.0),
            target_type="position",
        ),
    )

    converter = UrdfConverter(cfg)
    print(f"CONVERTED_USD: {converter.usd_path}")

    # Report the articulation the converter actually produced, to confirm the Revo2 joints survived
    # and to see exactly which DoFs have to be driven. Expect 41: 29 body + 6 per hand.
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(converter.usd_path)
    joints = [
        prim.GetName()
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    ]
    hand_kw = ("thumb", "index", "middle", "ring", "pinky")
    hand_joints = sorted(j for j in joints if any(k in j.lower() for k in hand_kw))

    print(f"TOTAL_ARTICULATED_JOINTS: {len(joints)}")
    print(f"BODY_JOINTS: {len(joints) - len(hand_joints)}")
    print(f"HAND_JOINTS ({len(hand_joints)}):")
    for j in hand_joints:
        print(f"  {j}")


if __name__ == "__main__":
    main()
    simulation_app.close()
