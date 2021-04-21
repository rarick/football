import click


@click.command()
@click.argument('config_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
def main(config_path):
  '''Run PartyFootball using a YAML config file'''
  print(config_path)


if __name__ == '__main__':
  main()
