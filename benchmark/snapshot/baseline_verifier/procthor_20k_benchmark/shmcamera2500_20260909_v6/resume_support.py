"""Outcome-independent recovery: immutable finished results, archive interruptions."""
import json
from pathlib import Path
import time
from protocol import validate_result


def completed_or_archive(directory, method, task):
    directory=Path(directory)
    result=directory/'result.json'
    if result.exists():
        row=json.loads(result.read_text())
        validate_result(row, method, task)
        if not (directory/'trajectory.npz').is_file():
            raise RuntimeError(f'Completed result missing trajectory: {result}')
        return row  # Includes failures: never retry them to improve success rate.
    if directory.exists():
        archived=directory.parent.parent/'interrupted_episodes'
        archived.mkdir(exist_ok=True)
        directory.rename(archived/f'{directory.parent.name}_{method}_{time.time_ns()}')
    return None
