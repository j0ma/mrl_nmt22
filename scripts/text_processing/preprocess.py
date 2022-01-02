import mrl_nmt.preprocessing as pp
import click


@click.command()
@click.option("--toml-config", type=click.Path(exists=True))
@click.option("--yaml-config", type=click.Path(exists=True))
@click.option("--verbose", is_flag=True)
def main(toml_config, yaml_config, verbose):

    assert bool(toml_config) ^ bool(
        yaml_config
    ), "Can only read TOML or YAML, not both."

    if toml_config:
        pipeline = pp.ExperimentPreprocessingPipeline.from_toml(
            toml_config, verbose=verbose
        )
    else:
        pipeline = pp.ExperimentPreprocessingPipeline.from_yaml(
            yaml_config, verbose=verbose
        )

    pipeline.process()


if __name__ == "__main__":
    main()
