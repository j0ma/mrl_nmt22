import mrl_nmt.preprocessing as pp
import click


@click.command()
@click.option("--toml-config", required=True, type=click.Path(exists=True))
@click.option("--verbose", is_flag=True)
def main(toml_config, verbose):
    pipeline = pp.ExperimentPreprocessingPipeline.from_toml(
        toml_config, verbose=verbose
    )
    pipeline.process()


if __name__ == "__main__":
    main()
