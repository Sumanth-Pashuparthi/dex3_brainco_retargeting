"""Isaac Lab Mimic plug-in for the Unitree G1 static apple-to-plate task.

Registered through Arena's ``--external_environment_class_path`` hook, so no Arena source is
modified::

    --external_environment_class_path g1_apple_mimic.environment:GalileoG1StaticAppleMimicEnvironment

Environment name: ``galileo_g1_static_apple_mimic`` (same scene, embodiment, spawn poses and
success term as ``galileo_g1_static_pick_and_place``; only adds the Mimic subtask graph and the
automatic ``grasp_<arm>`` subtask-termination signal).

Embodiments: Arena's ``g1_wbc_agile_pink`` (Dex3) or ``g1_wbc_agile_pink_brainco`` (BrainCo Revo2 hands,
defined in :mod:`g1_apple_mimic.brainco_pink`, registered when the environment module is imported).
"""
