#!/usr/bin/env python3

import click

from partyball.party_config import PartyConfig
from partyball.party_env import PartyEnv


@click.command()
@click.argument('config_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
def main(config_path):
  '''Run PartyFootball using a YAML config file'''

  party_config = PartyConfig.from_yaml(config_path)
  env = PartyEnv(party_config)

  # ----> Now do some work


if __name__ == '__main__':
  main()
