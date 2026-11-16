from .vdp import VanDerPol
from .slow_manifold import SlowManifold
from .manipulator import Manipulator
from .cartpole import Cartpole

SYSTEMS = {
    "vdp": VanDerPol,
    "slow_manifold": SlowManifold,
    "manipulator": Manipulator,
    "cartpole": Cartpole,
}


def get_system(name, **kwargs):
    if name in ("mujoco_ip", "mujoco_idp"):
        # imported lazily so the analytic systems work without mujoco installed
        from . import mujoco_envs
        return mujoco_envs.get(name, **kwargs)
    return SYSTEMS[name](**kwargs)


def experiment_system(name, **kwargs):
    """The system as used by the newer experiment scripts. Same as get_system,
    except that the inverted pendulum maps theta onto the gain box its model was
    trained with (Table III used a wider box, see mujoco_envs.py)."""
    if name == "mujoco_ip":
        kwargs.setdefault("gain_box", "training")
    return get_system(name, **kwargs)
