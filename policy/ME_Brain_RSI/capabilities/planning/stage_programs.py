"""Fixed upstream memory program through the exact released shared executor."""
import hashlib
import os
from pathlib import Path

MEMORY_SHA256 = 'edca144a6b78fb25484713da3ac8bd3763198beab8a293432f9c9609cd627919'


def execute(owner, source_file, program, arguments):
    # An ambient agent launch plan must never replace this frozen program.
    if any(os.environ.get(k) for k in ('LXM6WUEC_OPERATION_PLAN', 'LXM6WUEC_OPERATION_PLAN_SHA256')):
        raise ValueError('External operation plans are disabled for the fixed nesting policy')
    path = Path(source_file).with_suffix('.memory.json')
    if hashlib.sha256(path.read_bytes()).hexdigest() != MEMORY_SHA256:
        raise ValueError('Nesting operation memory changed')
    from lxm6wuec_skill_library.stage_programs import execute as original_execute
    return original_execute(owner, source_file, program, arguments)
