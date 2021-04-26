from gfootball.env.football_env_core import FootballEnvCore

from partyball.party_env import PartyEnv
from partyball.party_env_core import PartyEnvCore
from partyball.party_config import PartyConfig


def test_party_env_core():
    config = PartyConfig.from_yaml('../configs/default.yaml')
    env = PartyEnv(config)
    assert isinstance(env._env, PartyEnvCore)


def test_gfootball_env_core():
    config = PartyConfig.from_yaml('../configs/default.yaml')
    config['environment_core'] = 'gfootball'
    env = PartyEnv(config)
    assert isinstance(env._env, FootballEnvCore)
