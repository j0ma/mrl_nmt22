from pathlib import Path

from tqdm import tqdm
import click

MAX_DEPTH=10

def rm_tree(pth):
    """Source: https://stackoverflow.com/questions/50186904/pathlib-recursively-remove-directory"""
    pth = Path(pth)
    print(f"[rm_tree] Recursively removing {pth}")
    for child in pth.glob("*"):
        if child.is_symlink():
            rm_tree(child.resolve().expanduser())
        elif child.is_file():
            child.unlink(missing_ok=True)
        else:
            rm_tree(child)
    try:
        pth.rmdir()
    except NotADirectoryError:
        pth.unlink(missing_ok=True)


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
            if p.is_dir:
                rm_tree(p)
            else:
                p.unlink(missing_ok=True)

    # remove experiment folder itself
    print(f"Removing top-level experiment folder: {exp_path}")
    exp_path.rmdir()


if __name__ == "__main__":
    main()
