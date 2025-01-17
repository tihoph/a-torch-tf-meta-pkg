"""Creates git tags for each working combination."""

# ruff: noqa: S603, E501
from __future__ import annotations

import itertools
import os
import shutil
import subprocess as sp
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2
from pip._vendor.packaging.version import Version
from tqdm import tqdm

from setup import MetaDef

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def chdir(path: str | Path) -> Generator[None]:
    """Changes temporarily the current directory."""
    cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def create_git_tag(meta_def: MetaDef) -> None:
    """Creates a git tag with the meta definition."""
    if bool(meta_def.errors):
        return
    git_bin = shutil.which("git")
    if not git_bin:
        raise FileNotFoundError("could not find git.")
    short_tf = [str(x) for x in meta_def.tf.release[:2]]
    version = f"{''.join(short_tf)}.{''.join(str(x) for x in meta_def.torch.release)}"
    commit_msg = f"TensorFlow {'.'.join(short_tf)}, PyTorch {meta_def.torch}"
    setup_path = Path("setup.py")
    files = "ttest_loader.py", "ttest_matrix.py", "ttest_tag.py"
    temp_branch = str(uuid.uuid4())
    sp.check_call([git_bin, "checkout", "-b", temp_branch])
    sp.check_call([git_bin, "rm", "--cached", *files])
    setup_content = setup_path.read_text()
    fixed_setup = jinja2.Template(setup_content).render(
        tf_version=meta_def.tf, torch_version=meta_def.torch
    )
    setup_path.write_text(fixed_setup)
    for file in files:
        Path(file).unlink()
    sp.check_call([git_bin, "add", str(setup_path)])
    sp.check_call([git_bin, "commit", "-m", commit_msg])
    sp.check_call([git_bin, "tag", "-f", f"v{version}", temp_branch])
    sp.check_call([git_bin, "push", "-f", "origin", f"v{version}"])


if __name__ == "__main__":
    meta_defs: list[MetaDef] = []

    tf_versions = [Version(f"2.{minor}.0") for minor in range(10, 19)]
    torch_versions = [Version(x) for x in ("2.0.1", "2.3.1", "2.5.1")]

    for tf, torch in itertools.product(tf_versions, torch_versions):
        meta_def = MetaDef(Version("3.10"), tf, torch)
        meta_defs.append(meta_def)

    main_dir = Path().cwd()
    for meta_def in tqdm(meta_defs):
        with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
            shutil.copytree(main_dir, tmp, dirs_exist_ok=True)
            create_git_tag(meta_def)
