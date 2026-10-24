"""What is needed to reproduce a run: package and library versions, git state, host, input file fingerprints."""
from __future__ import annotations
import datetime, glob, hashlib, os, platform, subprocess, sys


def file_fingerprint(path, chunk=16 << 20):
    """Size, modification time and sha256 of the first and last 16 MB (fast for multi-GB ROOT files)."""
    st = os.stat(path); h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
        h.update(f.read(chunk))
    return dict(path=os.path.abspath(path), size=st.st_size,
                mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                sha256_head_tail=h.hexdigest())


def expand(paths):
    return [f for p in paths for f in (sorted(glob.glob(p)) or [p])]


def git_state():
    here = os.path.dirname(os.path.abspath(__file__))
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = run("rev-parse", "HEAD")
    return dict(commit=commit or None, dirty=bool(run("status", "--porcelain", "--untracked-files=no")) if commit else None)


def collect(cfg):
    from . import __version__
    libs = {}
    for m in ("numpy", "iminuit", "uproot", "awkward", "matplotlib", "yaml"):
        try:
            libs[m] = __import__(m).__version__
        except Exception:
            libs[m] = None
    return dict(
        time=datetime.datetime.now().isoformat(timespec="seconds"),
        scalesmear=__version__, git=git_state(), python=sys.version.split()[0], libraries=libs,
        host=platform.node(), platform=platform.platform(), cpu_count=os.cpu_count(),
        slurm_job=os.environ.get("SLURM_JOB_ID"), command=" ".join(sys.argv),
        inputs={k: [file_fingerprint(f) for f in expand(cfg["inputs"][k]) if os.path.exists(f)] for k in ("data", "mc")},
    )
