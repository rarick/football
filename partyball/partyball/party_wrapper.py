# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Environment that can be used with OpenAI Baselines."""

import gym
import numpy as np


class PartyObservationWrapper(gym.ObservationWrapper):

  def __init__(self, env, fixed_positions=False):
    gym.ObservationWrapper.__init__(self, env)
    action_shape = np.shape(self.env.action_space)
    shape = (action_shape[0] if len(action_shape) else 1, 900)
    self.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    self._fixed_positions = fixed_positions

  def observation(self, observation):
    return PartyObservationWrapper.convert_observation(observation, self._fixed_positions)

  @staticmethod
  def create_rel_pose(ref_pos_array, ref_active, rel_pos_array, rel_active):
      rel_pose_array = []
      for i in range(11):
        for j in range(11):
          rel_pos = None
          if not ref_active[i] or not rel_active[j]:
            rel_pos = np.zeros(2)  # Not sure how to encode this
          else:
            rel_pos = np.array(ref_pos_array[i]) - np.array(rel_pos_array[j])
          rel_pose_array.append(rel_pos)
      return rel_pose_array

  @staticmethod
  def do_flatten(obj):
      """Run flatten on either python list or numpy array."""
      if type(obj) == list:
        return np.array(obj).flatten()
      return obj.flatten()

  @staticmethod
  def convert_observation(observation, fixed_positions):
    final_obs = []
    for obs in observation:
      o = []
      if fixed_positions:
        for i, name in enumerate(['left_team', 'left_team_direction',
                                  'right_team', 'right_team_direction']):
          o.extend(PartyObservationWrapper.do_flatten(obs[name]))
          # If there were less than 11vs11 players we backfill missing values
          # with -1.
          if len(o) < (i + 1) * 22:
            o.extend([-1] * ((i + 1) * 22 - len(o)))
      else:
        o.extend(PartyObservationWrapper.do_flatten(obs['left_team']))
        o.extend(PartyObservationWrapper.do_flatten(obs['left_team_direction']))
        o.extend(PartyObservationWrapper.do_flatten(obs['right_team']))
        o.extend(PartyObservationWrapper.do_flatten(obs['right_team_direction']))

      # If there were less than 11vs11 players we backfill missing values with
      # -1.
      # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
      if len(o) < 88:
        o.extend([-1] * (88 - len(o)))

      # ball position
      o.extend(obs['ball'])
      # ball direction
      o.extend(obs['ball_direction'])
      # one hot encoding of which team owns the ball
      if obs['ball_owned_team'] == -1:
        o.extend([1, 0, 0])
      if obs['ball_owned_team'] == 0:
        o.extend([0, 1, 0])
      if obs['ball_owned_team'] == 1:
        o.extend([0, 0, 1])

      active = [0] * 11
      if obs['active'] != -1:
        active[obs['active']] = 1
      o.extend(active)

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)

      yellow_cards_left = obs['left_team_yellow_card']
      o.extend(yellow_cards_left)

      yellow_cards_right = obs['right_team_yellow_card']
      o.extend(yellow_cards_right)

      left_team_tired_factor = obs['left_team_tired_factor']
      o.extend(left_team_tired_factor)

      right_team_tired_factor = obs['right_team_tired_factor']
      o.extend(right_team_tired_factor)

      if obs['active'] != -1:
        active_player_pose = np.array(obs['left_team'][obs['active']])
        ball_pose =np.array(obs['ball'][:-1])
        rel_position = active_player_pose - ball_pose
        o.extend(rel_position)

      rel_pos_left_left = PartyObservationWrapper.create_rel_pose(
              obs['left_team'], obs['left_team_active'],
              obs['left_team'], obs['left_team_active'])

      rel_pos_left_right = PartyObservationWrapper.create_rel_pose(
              obs['left_team'], obs['left_team_active'],
              obs['right_team'], obs['right_team_active'])

      rel_pos_right_right = PartyObservationWrapper.create_rel_pose(
              obs['right_team'], obs['right_team_active'],
              obs['right_team'], obs['right_team_active'])

      o.extend(PartyObservationWrapper.do_flatten(rel_pos_left_left))
      o.extend(PartyObservationWrapper.do_flatten(rel_pos_left_right))
      o.extend(PartyObservationWrapper.do_flatten(rel_pos_right_right))

      sticky = obs['sticky_actions']
      o.extend(sticky)

      final_obs.append(o)

    try:
      return np.array(final_obs, dtype=np.float32)
    except Exception as e:
      for i, o in enumerate(final_obs):
        print('# {} all not list: {}'.format(i,
          all([type(a) is not list for a in o])))
      raise e
