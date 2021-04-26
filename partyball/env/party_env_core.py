from gfootball.env.football_env_core import FootballEnvCore


class PartyEnvCore(FootballEnvCore):

  _pass_actions = frozenset((9, 10 ,11))

  def __init__(self, config):
    super().__init__(config)
    self.accumulated_passes = 0

  def compute_reward(self, action):
    reward = 0
    action_cutoff = len(action) // 2
    left_actions = action[:action_cutoff]
    right_actions = action[action_cutoff:]

    curr_owned = self._observation['ball_owned_team']
    prev_owned = self._state.prev_ball_owned_team

    if (current_owned != -1 and
        prev_owned != -1 and
        current_owned != prev_owned):
      self.accumulated_passes = 0

    if curr_owned == 0:
      reward += 0.001
      if (left_actions[self._observation['ball_owned_player']] in
          PartyEnvCore._pass_actions):
        self.accumulated_passes += 1
    elif curr_owned == 1:
      reward -= 0.001
      if (right_actions[self._observation['ball_owned_player']] in
          PartyEnvCore._pass_actions):
        self.accumulated_passes += 1

    score_diff = self._observation['score'][0] - self._observation['score'][1]
    goal_diff = score_diff - self._state.previous_score_diff

    reward += goal_diff + goal_diff*self.accumulated_passes*0.1

    return reward
