"""Fetch pinned sources only for historical differential fixtures."""

import hashlib
import shutil
import tarfile
import urllib.request


def fetch_extract(spec, work):
    """Verify the complete upstream archive before extracting build inputs."""
    archive = work / "downloads" / spec["url"].rsplit("/", 1)[-1]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(spec["url"]) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch for {archive}: {actual}")
    with tarfile.open(archive) as source:
        root = work / source.getnames()[0].split("/", 1)[0]
        if not root.exists():
            source.extractall(work, filter="data")
    return root
