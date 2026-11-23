"""Load POSIX advisory locking only when a training run first acquires a lock."""

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        import fcntl
    except ModuleNotFoundError as exc:
        raise OSError("This trainer requires POSIX file locking. Run it on Linux.") from exc
    return getattr(fcntl, name)
