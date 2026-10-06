# Backward-compatibility shim: module moved to weaksup/ (2026-09-01
# refactor). Keep importing from here or from weaksup; both work.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import weaksup.witnessed_action_window_proposer_v1 as _mod
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('__')})

if __name__ == "__main__":
    raise SystemExit(_mod.main())
