from .vdp import VanDerPol
from .slow_manifold import SlowManifold, SlowManifoldU2
from .manipulator import Manipulator
from .cartpole import Cartpole

SYSTEMS = {
    "vdp": VanDerPol,
    "slow_manifold": SlowManifold,
    "slow_manifold_u2": SlowManifoldU2,
    "manipulator": Manipulator,
    "cartpole": Cartpole,
}


def get_system(name, **kwargs):
    if name in ("mujoco_ip", "mujoco_idp"):
        # imported lazily so the analytic systems work without mujoco installed
        from . import mujoco_envs
        return mujoco_envs.get(name, **kwargs)
    return SYSTEMS[name](**kwargs)
