from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path, PurePath
from typing import Literal, get_args
from urllib.parse import urlparse

import requests
from filelock import FileLock
from lamin_utils import logger
from requests.exceptions import RequestException
from rich.progress import Progress

COMPRESSION_FORMATS = Literal["tar.gz", "gztar", "zip", "tar", "gz", "bz", "xz"]
COMPRESSION_FORMATS_LIST = list(get_args(COMPRESSION_FORMATS))


def download(
    url: str,
    output_file_name: str | None = None,
    output_path: str | Path | None = None,
    archive_format: COMPRESSION_FORMATS | None = None,
    block_size: int = 1024,
    *,
    overwrite: bool = False,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> None | Path:  # pragma: no cover
    """Downloads a file irrespective of format.

    Args:
        url: URL to download.
        output_file_name: Name of the file to download. If not specified, the file name will be inferred from the URL.
        output_path: Path to download/extract the files to. Defaults to 'ehrapy_data/output_file_name' if not specified.
        archive_format: Format of the archive to download. If not specified, the format will be inferred from the URL.
        block_size: Block size for downloads in bytes.
        overwrite: Whether to overwrite existing files.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of download retries.
        retry_delay: Delay between retries in seconds.
    """
    raw_formats = ["csv", "txt", "parquet"]

    def _sanitize_file_name(file_name: str) -> str:
        if os.name == "nt":
            file_name = file_name.replace("?", "_").replace("*", "_")
        return file_name

    def _remove_archive_extension(file_name: str) -> str:
        for ext in COMPRESSION_FORMATS_LIST:
            if file_name.endswith(ext):
                return file_name.removesuffix(ext).rstrip(".")
        return file_name

    if output_path is None:
        output_path = tempfile.gettempdir()

    output_path = Path(output_path)

    url_file_name = PurePath(urlparse(url).path).name
    suffix = url_file_name.split(".")[-1]

    output_file_name = _sanitize_file_name(url_file_name) if output_file_name is None else output_file_name
    archive_format = suffix if archive_format is None else archive_format

    if archive_format in raw_formats:
        raw_data_output_path = output_path / output_file_name
        path_to_check = raw_data_output_path
    elif archive_format in COMPRESSION_FORMATS_LIST:
        tmpdir = tempfile.mkdtemp()
        raw_data_output_path = Path(tmpdir) / output_file_name
        path_to_check = output_path / _remove_archive_extension(output_file_name)
    else:
        msg = f"Unknown file format: {archive_format}"
        raise RuntimeError(msg)

    lock_path = f"{path_to_check}.lock"
    with FileLock(lock_path, timeout=300):
        if path_to_check.exists():
            warning = f"File {path_to_check} already exists!"
            if not overwrite:
                logger.warning(f"{warning} Using already downloaded dataset...")
                return path_to_check
            else:
                logger.warning(f"{warning} Overwriting...")

        temp_file_name = f"{raw_data_output_path}.part"

        retry_count = 0
        while retry_count < max_retries:
            try:
                head_response = requests.head(url, timeout=timeout)
                head_response.raise_for_status()
                content_length = int(head_response.headers.get("content-length", 0))
                free_space = shutil.disk_usage(output_path).free

                if content_length > free_space:
                    msg = f"Insufficient disk space. Need {content_length} bytes, but only {free_space} available."
                    raise OSError(msg)

                response = requests.get(url, stream=True)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                with Progress(refresh_per_second=5) as progress:
                    task = progress.add_task("[red]Downloading...", total=total)
                    with Path(temp_file_name).open("wb") as file:
                        for data in response.iter_content(block_size):
                            file.write(data)
                            progress.update(task, advance=len(data))
                        progress.update(task, completed=total, refresh=True)

                Path(temp_file_name).replace(raw_data_output_path)

                if archive_format in COMPRESSION_FORMATS_LIST:
                    shutil.unpack_archive(raw_data_output_path, output_path)

                return path_to_check

            except (OSError, RequestException) as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(
                        f"Download attempt {retry_count}/{max_retries} failed: {e!s}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Download failed after {max_retries} attempts: {e!s}")
                    if Path(temp_file_name).exists():
                        Path(temp_file_name).unlink(missing_ok=True)
                    raise

            except Exception as e:
                logger.error(f"Download failed: {e!s}")
                if Path(temp_file_name).exists():
                    Path(temp_file_name).unlink(missing_ok=True)
                raise
            finally:
                if Path(temp_file_name).exists():
                    Path(temp_file_name).unlink(missing_ok=True)
                Path(lock_path).unlink(missing_ok=True)

        return path_to_check
