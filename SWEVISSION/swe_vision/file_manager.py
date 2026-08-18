"""
File management for the Jupyter kernel's Docker container.

Handles copying files into the host-side mount directory so they
become accessible at /mnt/data inside the container.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Optional

from swe_vision.config import CONTAINER_WORK_DIR, HOST_WORK_DIR, logger


@dataclass
class NotebookFileManager:
    """
    Manages files that should be accessible in the Jupyter kernel's
    Docker container.

    Files are copied to the **host-side** mount directory, which is
    volume-mounted into the container at /mnt/data.  The model is told
    to reference files using the container-side path (/mnt/data/<name>).
    """
    host_work_dir: str = HOST_WORK_DIR
    container_work_dir: str = CONTAINER_WORK_DIR

    def setup_work_dir(
        self,
        host_work_dir: Optional[str] = None,
        container_work_dir: Optional[str] = None,
    ):
        if host_work_dir:
            self.host_work_dir = host_work_dir
        if container_work_dir:
            self.container_work_dir = container_work_dir
        os.makedirs(self.host_work_dir, exist_ok=True)
        logger.info(
            "NotebookFileManager: host_work_dir=%s, container_work_dir=%s",
            self.host_work_dir, self.container_work_dir,
        )

    def copy_file_to_workdir(self, src_path: str, dest_name: Optional[str] = None) -> str:
        """
        Copy a file into the host mount directory so the container kernel
        can access it at /mnt/data/<filename>.
        Returns the **container-side** path for use in prompts / hints.
        """
        if dest_name is None:
            dest_name = os.path.basename(src_path)
        os.makedirs(self.host_work_dir, exist_ok=True)
        host_dest = os.path.join(self.host_work_dir, dest_name)
        if os.path.abspath(src_path) != os.path.abspath(host_dest):
            shutil.copy2(src_path, host_dest)
            logger.info("Copied %s -> %s (container: %s)", src_path, host_dest,
                        os.path.join(self.container_work_dir, dest_name))
        return os.path.join(self.container_work_dir, dest_name)

    def get_kernel_path(self, filename: str) -> str:
        """Return the full path a file would have inside the container."""
        return os.path.join(self.container_work_dir, filename)
