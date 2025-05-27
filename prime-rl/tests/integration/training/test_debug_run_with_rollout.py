from pathlib import Path
from subprocess import Popen
from typing import Callable

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "torchrun", "src/zeroband/train.py", "@configs/training/debug.toml"]


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_rollout_run")


@pytest.fixture(scope="module")
def process(
    output_path: Path, run_process: Callable[[list[str]], Popen], fake_rollout_files_dir: Callable[[list[int], int, int, int], Path]
):
    data_path = fake_rollout_files_dir(steps=list(range(2)), num_files=8, batch_size=8, seq_len=16)
    return run_process(CMD + ["--data.path", str(data_path), "--no-data.fake"])


def test_no_error(process: Popen):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"
