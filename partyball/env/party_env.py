from gfootball.env.football_env import FootballEnvBase
from gfootball.env.football_env_core import FootballEnvCore
from partyball.env.party_env_core import PartyEnvCore


class PartyEnv(FootballEnvBase):

  def __init__(self, config, wrapper):
    super().__init__(config)
    self._env = PartyEnv._get_environment_core(
            self._config['environment_core'])(self._config)
    self.obs_wrapper = wrapper(self)

  _environment_cores = {
      'gfootball': FootballEnvCore,
      'party': PartyEnvCore,
  }

  def _get_environment_core(core_type):
    if core_type not in PartyEnv._environment_cores:
      raise Exception(f'Invalid environment_core: {core_type} . ' +
          'Expected one of: [{}]'.format(
              ', '.join(PartyEnv._environment_cores.keys())))

    return PartyEnv._environment_cores[core_type]
  def reset(self):
      self._env.reset()
      return self.obs_wrapper.observation([self._env._observation])

  def step(self, action):
      action = self._action_to_list(action)
      if self._agent:
          # self._agent.set_action(action)
        pass
      else:
          assert len(
              action
          ) == 0, 'step() received {} actions, but no agent is playing.'.format(
              len(action))

      # _, reward, done, info = self._env.step(self._get_actions())
      raw_obs, raw_reward, done, info = self._env.step(action)

      raw_obs = self.obs_wrapper.observation([raw_obs])
      return raw_obs, raw_reward, done, info