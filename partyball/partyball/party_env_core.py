from gfootball.env.football_env_core import FootballEnvCore


class PartyEnvCore(FootballEnvCore):

  _pass_actions = frozenset(('long_pass', 'high_pass', 'short_pass'))

  def __init__(self, config):
    super().__init__(config)
    self.accumulated_passes = 0
    self._prev_ball_owned_player = None
    self.max_dist_to_net = 2.04362423 #sqrt([1 - (-1)]**2 + [0.42 - 0]**2)

  def compute_reward(self, action):
    reward = 0

    curr_owned_team = self._observation['ball_owned_team']
    prev_owned_team = self._state.prev_ball_owned_team

    curr_owned_player = self._observation['ball_owned_player']
    prev_owned_player = self._prev_ball_owned_player
    self._prev_ball_owned_player = curr_owned_player

    # Possession changed
    if (curr_owned_team != -1 and
        prev_owned_team != -1 and
        curr_owned_team != prev_owned_team):
      self.accumulated_passes = 0

    # Someone has possession
    if curr_owned_team != -1:

      if curr_owned_team != prev_owned_team:
        self.accumulated_passes = 0
      elif curr_owned_player != prev_owned_player:
        # Reward a pass
        self.accumulated_passes += 1
        if curr_owned_team == 0:
          reward += 0.01
        elif curr_owned_team == 1:
          reward -= 0.01

      # Reward possession and shots according to distance to the opponent's net
      if curr_owned_team == 0:
        reward += 0.001
        if action[0]._name == "shot":
          player_coords = self._observation["left_team"][curr_owned_player]
          dist_to_net = ((player_coords[0] - 1)**2 + (player_coords[1] - 0)**2)**(.5)
          norm_dist_to_net = (1 - dist_to_net / self.max_dist_to_net)**2
          reward += 0.1*norm_dist_to_net


      elif curr_owned_team == 1:
        reward -= 0.001
        if action[1]._name == "shot":
          player_coords = self._observation["right_team"][curr_owned_player]
          dist_to_net = ((player_coords[0] + 1) ** 2 + (player_coords[1] - 0) ** 2) ** (.5)
          norm_dist_to_net = (1 - dist_to_net / self.max_dist_to_net)**2
          reward -= 0.1*norm_dist_to_net


    score_diff = self._observation['score'][0] - self._observation['score'][1]
    goal_diff = score_diff - self._state.previous_score_diff

    # Someone scored
    reward += goal_diff + goal_diff*self.accumulated_passes*0.1

    return reward
