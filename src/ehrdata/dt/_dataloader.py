from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal, get_args
from urllib.parse import urlparse

import pooch
from rich.console import Console
from rich.progress import Progress

from ehrdata._logger import logger

if TYPE_CHECKING:
    from rich.progress import TaskID

pooch.get_logger().setLevel(logging.WARNING)

COMPRESSION_FORMATS = Literal["tar.gz", "gztar", "zip", "tar", "gz", "bz", "xz"]
COMPRESSION_FORMATS_LIST = list(get_args(COMPRESSION_FORMATS))
RAW_FORMATS = Literal["csv", "txt", "parquet", "h5ad", "zarr"]
RAW_FORMATS_LIST = list(get_args(RAW_FORMATS))


class _RichProgress:
    """Adapter exposing the tqdm-like interface pooch expects, backed by rich.progress."""

    def __init__(self, description: str = "[red]Downloading..."):
        self._description = description
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self.total: int | None = None

    def _ensure_started(self) -> None:
        if self._progress is None:
            self._progress = Progress(refresh_per_second=3, console=Console(force_jupyter=False))
            self._progress.start()
            self._task = self._progress.add_task(self._description, total=self.total or None)

    def update(self, n: int) -> None:
        self._ensure_started()
        if self.total is not None and self._progress.tasks[self._task].total != self.total:
            self._progress.update(self._task, total=self.total or None)
        self._progress.update(self._task, advance=n)

    def reset(self) -> None:
        self._ensure_started()
        self._progress.reset(self._task, total=self.total or None)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None


def _download(
    url: str,
    output_filename: str | None = None,
    output_path: str | Path | None = None,
    archive_format: COMPRESSION_FORMATS | None = None,
    raw_format: RAW_FORMATS | None = None,
    block_size: int = 1024,
    *,
    overwrite: bool = False,
    timeout: int = 60,
) -> None | Path:  # pragma: no cover
    """Downloads a file irrespective of format.

    The download itself, including retries and caching, is delegated to
    `pooch <https://www.fatiando.org/pooch/>`_, in line with the scverse ecosystem.

    Args:
        url: URL to download.
        output_filename: Name of the file to download. If not specified, the file name will be inferred from the URL.
        output_path: Path to download/extract the files to. Defaults to 'ehrapy_data/output_filename' if not specified.
        archive_format: Format of the archive to download. If not specified, the format will be inferred from the URL.
        raw_format: Format of the raw file to download. If not specified, the format will be inferred from the URL.
        block_size: Block size for downloads in bytes.
        overwrite: Whether to overwrite existing files.
        timeout: Request timeout in seconds.
    """

    def _sanitize_filename(filename: str) -> str:
        if os.name == "nt":
            filename = filename.replace("?", "_").replace("*", "_")
        return filename

    def _remove_archive_extension(filename: str) -> str:
        for ext in COMPRESSION_FORMATS_LIST:
            if filename.endswith(ext):
                return filename.removesuffix(ext).rstrip(".")
        return filename

    if output_path is None:
        output_path = tempfile.gettempdir()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    url_filename = PurePath(urlparse(url).path).name
    suffix = url_filename.split(".")[-1]

    output_filename = _sanitize_filename(url_filename) if output_filename is None else output_filename

    if raw_format is None:
        file_ending = suffix
    elif archive_format is None:
        file_ending = raw_format
    else:
        file_ending = suffix

    if file_ending in RAW_FORMATS_LIST:
        download_dir = output_path
        raw_data_output_path = output_path / output_filename
        path_to_check = raw_data_output_path
    elif file_ending in COMPRESSION_FORMATS_LIST:
        download_dir = Path(tempfile.mkdtemp())
        raw_data_output_path = download_dir / output_filename
        path_to_check = output_path / _remove_archive_extension(output_filename)
    else:
        msg = f"Unknown file format: {file_ending}"
        raise RuntimeError(msg)

    if path_to_check.exists():
        warning = f"File {path_to_check} already exists!"
        if not overwrite:
            return path_to_check
        logger.warning(f"{warning} Overwriting...")
        # pooch does not re-fetch an existing file when no hash is given, so remove it to force a download.
        if raw_data_output_path.exists():
            raw_data_output_path.unlink()

    pooch.retrieve(
        url=url,
        known_hash=None,
        fname=output_filename,
        path=str(download_dir),
        downloader=pooch.HTTPDownloader(
            progressbar=_RichProgress(),
            chunk_size=block_size,
            timeout=timeout,
            headers={"User-Agent": "ehrdata/1.0.0 (https://github.com/theislab/ehrdata)"},
        ),
    )

    if file_ending in COMPRESSION_FORMATS_LIST:
        shutil.unpack_archive(raw_data_output_path, output_path)

    return path_to_check
