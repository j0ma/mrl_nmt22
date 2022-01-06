from pathlib import Path

import mrl_nmt.utils as u
from tqdm import tqdm
import click

MAX_DEPTH = 10


@click.command()
@click.option("--experiment-name", "-n")
@click.option("--prefix", "-p", default="")
@click.option("--keep-checkpoints", is_flag=True)
def main(experiment_name, prefix, keep_checkpoints):
    exp_path = Path(prefix).expanduser().absolute() / "experiments" / experiment_name

    # remove symlinks to bin and raw data
    data = exp_path / "raw_data"
    bin_data = exp_path / "binarized_data"

    print(f"Removing symlink for data: {data}")
    data.unlink(missing_ok=True)
    print(f"Removing symlink for bin data: {bin_data}")
    bin_data.unlink(missing_ok=True)

    if keep_checkpoints:
        pass  # TODO: implement

    # remove all other model folders
    rest = exp_path.glob("*")
    if rest:
        for fname in tqdm(rest):
            p = Path(fname).expanduser().absolute()
            print(f"Removing other folder/file: {p}")
            if p.is_dir():
                u.recursively_delete(p)
            else:
                p.unlink(missing_ok=True)

    # remove experiment folder itself
    print(f"Removing top-level experiment folder: {exp_path}")
    exp_path.rmdir()


if __name__ == "__main__":
    main()
