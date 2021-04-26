from gfootball.env.football_env_core import FootballEnvCore


class PartyEnvCore(FootballEnvCore):
	def __init__(self, config):
		super().__init__(config)
		self.accumulated_passes = 0

	def compute_reward(self, action):
		reward = 0
		left_actions = action[:int(len(action)/2)]
		right_actions = action[-int(len(action)/2):]
		if self._observation['ball_owned_team'] != -1 and
        self._state.prev_ball_owned_team != -1 and
        self._observation['ball_owned_team'] !=
        self._state.prev_ball_owned_team:
        	self.accumulated_passes = 0 
        if self._observation['ball_owned_team'] == 0:
        	if left_actions[self._observation['ball_owned_player']] in {9,10,11}:
        		self.accumulated_passes += 1
        if self._observation['ball_owned_team'] == 1:
        	if right_actions[self._observation['ball_owned_player']] in {9,10,11}:
        		self.accumulated_passes += 1
		if self._observation['ball_owned_team'] == 0:
			reward += 0.001
		else:
			reward -= 0.001
    score_diff = self._observation['score'][0] - self._observation['score'][1]
    goal_dif = score_diff - self._state.previous_score_diff
    reward += goal_dif + goal_dif*self.accumulated_passes*0.1
    return reward

    
    