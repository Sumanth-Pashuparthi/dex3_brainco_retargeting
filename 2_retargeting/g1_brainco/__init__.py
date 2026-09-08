# Registration happens in isaaclab_arena/embodiments/__init__.py, which imports g1_brainco.g1_brainco
# so that @register_asset runs. Kept empty deliberately: importing g1_brainco here would pull Isaac
# Lab in at package-import time and break the standalone retargeting test.
