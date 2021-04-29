from gfootball.env.football_env_core import FootballEnvCore


class PartyEnvCore(FootballEnvCore):

  _pass_actions = frozenset(('long_pass', 'high_pass', 'short_pass'))

  def __init__(self, config):
    super().__init__(config)
    self.accumulated_passes = 0
    self._prev_ball_owned_player = None

  def compute_reward(self, action):
    reward = 0

    curr_owned_team = self._observation['ball_owned_team']
    prev_owned_team = self._state.prev_ball_owned_team

    curr_owned_player = self._observation['ball_owned_player']
    prev_owned_player = self._prev_ball_owned_player
    self._prev_ball_owned_player = curr_owned_player

    if (curr_owned_team != -1 and
        prev_owned_team != -1 and
        curr_owned_team != prev_owned_team):
      self.accumulated_passes = 0

    if curr_owned_team != -1:

      if curr_owned_team != prev_owned_team:
        self.accumulated_passes = 0
      elif curr_owned_player == prev_owned_player:
        self.accumulated_passes += 1

      if curr_owned_team == 0:
        reward += 0.001
      elif curr_owned_team == 1:
        reward -= 0.001

    score_diff = self._observation['score'][0] - self._observation['score'][1]
    goal_diff = score_diff - self._state.previous_score_diff

    reward += goal_diff + goal_diff*self.accumulated_passes*0.1

    return reward
