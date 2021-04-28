from partyball.env.party_config import PartyConfig
from partyball.env.party_wrapper import PartyObservationWrapper
from partyball.env.party_env import PartyEnv
from partyball.env.party_agent import PartyAgent

config = PartyConfig.from_yaml("partyball/configs/training.yaml")
env = PartyEnv(config, PartyObservationWrapper)
agent = PartyAgent(env)
agent.learn(1)

print(agent.eval(1))
