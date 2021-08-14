"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickPlaceEnvV2, SawyerCoffeePullEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v1 import SawyerCoffeePullEnv
from metaworld.policies.sawyer_coffee_pull_v2_policy import SawyerCoffeePullV2Policy
from metaworld.policies import SawyerCoffeePullV2Policy, SawyerPickPlaceV2Policy, SawyerCoffeePullV1Policy





# env = SawyerPickPlaceEnvV2()
# env = SawyerPickAndPlaceEnv()
env = SawyerCoffeePullEnv()
# env = SawyerCoffeePullEnvV2()
# import pdb; pdb.set_trace()
# policy = SawyerCoffeePullV2Policy()
# policy = SawyerPickPlaceV2Policy()
policy = SawyerCoffeePullV1Policy()

env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
env.reset()
env._freeze_rand_vec = True
lock_action = False
random_action = False
obs = env.reset()
action = np.zeros(4)
while True:
    # action = env.action_space.sample()
    action = policy.get_action(obs)
    ob, reward, done, infos = env.step(action)
    # time.sleep(1)
    if done:
        obs = env.reset()
    env.render()
