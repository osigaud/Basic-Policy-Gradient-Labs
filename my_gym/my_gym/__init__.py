
from gym.envs.registration import register

register(
    id='CartPoleContinuous-v0',
    entry_point='my_gym.envs:ContinuousCartPoleEnv',
)
