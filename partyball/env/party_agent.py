import numpy as np
from gym.spaces import Box
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


class PartyAgent:
    def __init__(self, env, model_verbose=1):

        self.env = env

        # Observation space is not initialized natively since it depends of the wrapper
        # We might be able to narrow the limits instead of using inf
        init_state = env.reset()
        n_obs = np.prod(init_state.shape)
        self.env.observation_space = Box(np.array([-np.inf]*n_obs), np.array([np.inf]*n_obs))

        self.model = PPO2(MlpPolicy, env, verbose=model_verbose)

    def learn(self, timesteps):
        self.model.learn(timesteps)

    def save_agent(self):
        # TODO:
        pass

    def load_agent(self):
        # TODO:
        pass

    def take_action(self, obs_state):
        action, _states = self.model.predict(obs_state)
        return action

    def _run_game(self, render=False):
        if render:
            self.env.render()

        obs = self.env.reset()
        done = False
        while not done:
            action = self.take_action(obs)
            obs, rewards, done, info = self.env.step(action)
        return self.env._env._observation["score"]

    def eval(self, n_evaluations=1):
        score_diffs = []
        for _ in range(n_evaluations):
            score = self._run_game(render=False)
            score_diffs.append(score[0]-score[1])
        return np.mean(score_diffs)
