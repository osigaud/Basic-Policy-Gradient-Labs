import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CartPoleContinuous-v0',
    entry_point='continuous_cartpole.envs:ContinuousCartPoleEnv',
)
