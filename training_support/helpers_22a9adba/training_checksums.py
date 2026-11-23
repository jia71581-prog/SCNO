"""Helpers used by the public training entry points."""
from hashlib import file_digest
from pathlib import Path

SHALLOW_SHA256 = "8321574485ebc7c5abcc937e0d61d7440a16802d39bb81739878a5e1f037e9b3"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
