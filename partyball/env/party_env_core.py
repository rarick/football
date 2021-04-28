from gfootball.env.football_env_core import FootballEnvCore


class PartyEnvCore(FootballEnvCore):

  _pass_actions = frozenset((9, 10 ,11))

  def __init__(self, config):
    super().__init__(config)
    self.accumulated_passes = 0
    self.possession_count = [0, 0]

  def compute_reward(self, action):
    reward = 0

    curr_owned = self._observation['ball_owned_team']

    if curr_owned != -1:
      self.possession_count[curr_owned] += 1
      self.possession_count[1 - curr_owned] = 0

    if curr_owned == 0:
      reward += 0.001 * self.possession_count[0]
    elif curr_owned == 1:
      reward -= 0.001 * self.possession_count[1]

    score_diff = self._observation['score'][0] - self._observation['score'][1]
    goal_diff = score_diff - self._state.previous_score_diff

    reward += goal_diff

    return reward
