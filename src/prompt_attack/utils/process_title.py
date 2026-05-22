"""Best-effort process title updates for tmux, ps, and nvidia-smi visibility."""

from __future__ import annotations


def set_process_title(title: str) -> bool:
    """Set the current process title when setproctitle is installed."""
    try:
        from setproctitle import setproctitle
    except ImportError:
        return False

    setproctitle(title[:255])
    return True
