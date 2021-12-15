from pathlib import Path

from tqdm import tqdm
import click


def rm_tree(pth):
    """Source: https://stackoverflow.com/questions/50186904/pathlib-recursively-remove-directory"""
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink(missing_ok=True)
        else:
            rm_tree(child)
    pth.rmdir()


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

    # remove all checkpoints unless told otherwise
    if not keep_checkpoints:
        checkpoints = exp_path / "checkpoints"
        checkpoints_abs = checkpoints.resolve().expanduser().absolute()
        try:
            print(f"Removing checkpoints: {checkpoints_abs}")
            checkpoints_abs.rmdir()
        except FileNotFoundError:
            pass
        checkpoints.unlink(missing_ok=True)

    # remove all other evaluation outputs
    rest = exp_path.glob("*")
    if rest:
        for fname in tqdm(rest):
            print(f"Removing other folder: {fname}")
            p = Path(fname).expanduser().absolute()
            if p.is_dir:
                rm_tree(p)
            else:
                p.unlink(missing_ok=True)

    # remove experiment folder itself
    print(f"Removing top-level experiment folder: {exp_path}")
    exp_path.rmdir()


if __name__ == "__main__":
    main()
