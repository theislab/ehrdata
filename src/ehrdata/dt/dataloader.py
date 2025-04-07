from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import requests
from filelock import FileLock

# from rich import print
from rich.progress import Progress


def download(
    url: str,
    saving_path: Path | str,
    block_size: int = 1024,
    overwrite: bool = False,
) -> None:  # pragma: no cover
    """Downloads a file irrespective of format.

    Args:
        url: URL to download.
        download_path: Where the data should be downloaded to.
    """
    # note: tar.gz has to be before gz for the _remove_archive_extension function to remove the entire extension
    compression_formats = ["tar.gz", "zip", "tar", "gz", "bz", "xz"]
    raw_formats = ["csv", "txt", "parquet"]

    saving_path = Path(saving_path)
    # urls can end with "?download"
    file_name = os.path.basename(url).split("?")[0]
    suffix = file_name.split(".")[-1]

    def _remove_archive_extension(file_path: str) -> str:
        for ext in compression_formats:
            # if the file path ends with extension, remove the extension and the dot before it (hence the -1)
            if file_path.endswith(ext):
                return file_path[: -len(ext) - 1]
        return file_path

    if suffix in raw_formats:
        raw_data_saving_path = saving_path / file_name
        path_to_check = raw_data_saving_path
    elif suffix in compression_formats:
        tmpdir = tempfile.mkdtemp()
        raw_data_saving_path = Path(tmpdir) / file_name
        path_to_check = saving_path / _remove_archive_extension(file_name)
    else:
        raise RuntimeError(f"Unknown file format: {suffix}")
        return

    if path_to_check.exists():
        info = f"File {path_to_check} already exists!"
        if not overwrite:
            logging.info(f"{info} Use downloaded dataset...")
            return
        else:
            logging.info(f"{info} Overwriting...")

    logging.info(f"Downloading {file_name} from {url} to {raw_data_saving_path}")

    lock_path = f"{raw_data_saving_path}.lock"
    with FileLock(lock_path):
        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        temp_file_name = f"{raw_data_saving_path}.part"

        with Progress(refresh_per_second=1500) as progress:
            task = progress.add_task("[red]Downloading...", total=total)
            with Path(temp_file_name).open("wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=block_size)

            # force the progress bar to 100% at the end
            progress.update(task, completed=total, refresh=True)

            Path(temp_file_name).replace(raw_data_saving_path)

        if suffix in compression_formats:
            shutil.unpack_archive(raw_data_saving_path, saving_path)
            logging.info(
                f"Extracted archive {file_name} from {raw_data_saving_path} to {saving_path / _remove_archive_extension(file_name)}"
            )

    Path(lock_path).unlink(missing_ok=True)
