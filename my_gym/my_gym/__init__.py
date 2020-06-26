import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CartPoleContinuous-v0',
    entry_point='my_gym.envs:Continuous_CartPoleEnv',
)
