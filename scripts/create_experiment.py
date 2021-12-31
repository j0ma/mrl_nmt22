#!/usr/bin/env python

from typing import Optional, Union
from pathlib import Path
import subprocess as sp
import datetime
import tempfile as tf

import click
import attr
import mrl_nmt.utils as u

EXISTING_FOLDER_OR_FILE = click.Path(
    exists=True, dir_okay=True, file_okay=False, path_type=Path
)
EXISTING_FILE = click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path)
PATH_OR_STR = Union[Path, str]

default_experiment_path = Path("./experiments")
default_checkpoint_path = Path("./checkpoints")


@attr.s()
class ExperimentFolder:

    _tmp_prefix = "/tmp/__mrl"

    experiment_name: str = attr.ib()

    # overall prefix to create folders under
    prefix = attr.ib(default=default_experiment_path, converter=u.str_to_path)

    # overall prefix for creating checkpoints
    checkpoint_prefix = attr.ib(
        default=default_checkpoint_path, converter=u.str_to_path
    )

    def __attrs_post_init__(self):

        if not self.full_path.exists():
            self.full_path.mkdir(exist_ok=True, parents=True)

    @property
    def full_path(self) -> Path:
        return Path(f"{self.prefix.expanduser()}/{self.experiment_name}")

    @property
    def eval_path(self) -> Path:
        return self.full_path / "eval"

    @property
    def train_path(self) -> Path:
        return self.full_path / "train"

    def create_initial_folder(self) -> None:
        self.full_path.mkdir()

    def create_eval(
        self,
        eval_name: str,
        checkpoint: Path,
        clean_references_file: Path,
        bin_data_folder: Path,
        raw_data_folder: Path,
    ) -> Path:

        # we need everything to exist for this eval to work

        for path in checkpoint, clean_references_file, bin_data_folder:
            assert path.exists(), f"Path {path} does not exist!"

        eval_folder = self.eval_path / eval_name

        try:

            # step 1: create actual folder
            eval_folder.mkdir(parents=True, exist_ok=True)

            # step 2: link checkpoint
            u.create_symlink(
                link_path=(eval_folder / "checkpoint"), dest_path=checkpoint
            )

            # step 3: link bin data
            u.create_symlink(
                link_path=(eval_folder / "binarized_data"), dest_path=bin_data_folder
            )

            # step 4: link references file
            u.create_symlink(
                link_path=(eval_folder / "clean_references"),
                dest_path=clean_references_file,
            )

        except (FileNotFoundError, OSError):
            self.tear_down_folder(path=eval_folder)

        return eval_folder

    def create_train(
        self,
        model_name: str,
        raw_data_folder: Path,
        bin_data_folder: Path,
        clean_references_file: Path,
        add_new_eval: bool = False,
    ) -> Path:

        # we need everything to exist for this eval to work

        for path in raw_data_folder, bin_data_folder, clean_references_file:
            assert path.exists(), f"Path {path} does not exist!"

        train_folder = self.train_path / model_name

        try:

            # step 0: create actual folder
            train_folder.mkdir(parents=True, exist_ok=True)

            # step 1: create checkpoint folder
            self.checkpoint_folder = self.create_checkpoint_folder(
                model_name=model_name
            )

            # step 2: link checkpoint
            u.create_symlink(
                link_path=(train_folder / "checkpoints"),
                dest_path=self.checkpoint_folder,
            )

            # step 3: link bin data
            u.create_symlink(
                link_path=(train_folder / "binarized_data"), dest_path=bin_data_folder
            )

            # step 4: link raw data
            u.create_symlink(
                link_path=(train_folder / "raw_data"), dest_path=raw_data_folder
            )

            # step 5: create & link eval if needed

            if add_new_eval:
                self.add_eval_to_train(
                    model_name=model_name,
                    eval_name=f"eval_{model_name}",
                    checkpoint=train_folder / "checkpoints" / "checkpoint_best.pt",
                    clean_references_file=clean_references_file,
                    raw_data_folder=raw_data_folder,
                    bin_data_folder=bin_data_folder,
                )

        except (FileNotFoundError, OSError):
            self.tear_down_folder(path=train_folder)
            self.tear_down_folder(path=self.checkpoint_folder)

        return train_folder

    ## TODO: deprecate this behavior
    def create_legacy_experiment(
        self,
        model_name: str,
        raw_data_folder: Path,
        bin_data_folder: Path,
        clean_references_file: Path,
        add_new_eval: bool = False,
    ) -> Path:

        # we need everything to exist for this eval to work

        for path in raw_data_folder, bin_data_folder, clean_references_file:
            assert path.exists(), f"Path {path} does not exist!"

        # step 0: link bin data
        u.create_symlink(
            link_path=(self.full_path / "binarized_data"), dest_path=bin_data_folder
        )

        # step 1: link raw data
        u.create_symlink(
            link_path=(self.full_path / "raw_data"), dest_path=raw_data_folder
        )

        eval_folder = self.full_path / model_name

        try:
            # step 2: create actual folder for eval
            eval_folder.mkdir(parents=True, exist_ok=True)

            # step 3: create checkpoint folder
            self.checkpoint_folder = self.create_checkpoint_folder(
                model_name=model_name
            )

            # step 4: link checkpoint
            u.create_symlink(
                link_path=(eval_folder / "checkpoints"),
                dest_path=self.checkpoint_folder,
            )

        except (FileNotFoundError, OSError):
            self.tear_down_folder(path=eval_folder)
            self.tear_down_folder(path=self.checkpoint_folder)

        return eval_folder

    def create_checkpoint_folder(self, model_name: str) -> Path:
        time_slug = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H%M")
        cpt_folder = f"{self.experiment_name}-{model_name}-{time_slug}"
        full_cpt_path = self.checkpoint_prefix / cpt_folder

        try:
            full_cpt_path.mkdir(parents=True, exist_ok=True)
        except (FileNotFoundError, OSError):
            self.tear_down_folder(full_cpt_path)

        return full_cpt_path

    def add_eval_to_train(
        self,
        model_name: str,
        eval_name: str,
        checkpoint: Path,
        raw_data_folder: Path,
        bin_data_folder: Path,
        clean_references_file: Path,
    ):
        for path in checkpoint, raw_data_folder, bin_data_folder, clean_references_file:
            assert path.exists(), f"Path {path} does not exist!"

        eval_folder_path = self.create_eval(
            eval_name=eval_name,
            checkpoint=checkpoint,
            clean_references_file=clean_references_file,
            bin_data_folder=bin_data_folder,
            raw_data_folder=raw_data_folder,
        )
        u.create_symlink(
            link_path=(self.train_path / model_name / eval_name),
            dest_path=eval_folder_path,
        )

    def tear_down_folder(self, path: PATH_OR_STR) -> None:
        sp.run(["rm", "-vrf", path])

    def teardown(self) -> None:
        self.tear_down_folder(self.full_path)


@click.command()
@click.option("--legacy-mode", is_flag=True)
@click.option("--experiment-name", required=True)
@click.option("--raw-data-folder", type=EXISTING_FOLDER_OR_FILE)
@click.option("--references-file", "--refs", type=EXISTING_FOLDER_OR_FILE)
@click.option(
    "--bin-data-folder",
    type=EXISTING_FOLDER_OR_FILE,
)
@click.option("--model-name", help="What to call the model")
@click.option(
    "--experiments-prefix",
    help="Folder to create experiments in",
    default="./experiments",
    type=EXISTING_FOLDER_OR_FILE,
)
@click.option(
    "--checkpoint-prefix",
    help="Folder to create checkpoints in",
    default="./checkpoints",
    type=EXISTING_FOLDER_OR_FILE,
)
@click.option("--eval-only", is_flag=True, help="Do not create a train/ subdirectory")
@click.option("--eval-name", help="What to call the eval run")
@click.option("--eval-model-checkpoint", default=EXISTING_FILE)
def main(
    legacy_mode,
    experiment_name,
    raw_data_folder,
    bin_data_folder,
    references_file,
    model_name,
    experiment_prefix,
    checkpoint_prefix,
    eval_only,
    eval_name,
    eval_model_checkpoint,
):

    ef = ExperimentFolder(
        experiment_name=experiment_name, checkpoint_prefix=checkpoint_prefix
    )

    if legacy_mode:
        ef.create_legacy_experiment(
            model_name=model_name,
            checkpoint=eval_model_checkpoint,
            raw_data_folder=raw_data_folder,
            bin_data_folder=bin_data_folder,
            clean_references_file=references_file,
        )
    elif eval_only:
        ef.create_eval(
            eval_name=eval_name,
            checkpoint=eval_model_checkpoint,
            raw_data_folder=raw_data_folder,
            bin_data_folder=bin_data_folder,
            clean_references_file=references_file,
        )
    else:
        ef.create_train(
            model_name=model_name,
            checkpoint=eval_model_checkpoint,
            raw_data_folder=raw_data_folder,
            bin_data_folder=bin_data_folder,
            clean_references_file=references_file,
        )


if __name__ == "__main__":
    main()
