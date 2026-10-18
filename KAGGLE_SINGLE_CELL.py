# =============================================================================
# Z-Image-Turbo stickman test — KAGGLE SINGLE CELL (paste this whole thing
# into ONE cell of a new Kaggle notebook and run it)
#
# Kaggle Settings first:  Accelerator = GPU T4 x2   |   Internet = ON
#
# What it does: GPU check -> installs diffusers(GitHub)+sdnq -> fixes the
# torchao<0.15 vs diffusers>=0.41 clash (FqnToConfig) with a self-healing
# preflight -> downloads z_image_turbo_test.py from the committed GitHub
# branch -> runs it (4 stickman images, 1024px) -> displays results.
# Everything runs in subprocesses, so no kernel restart is needed.
# First run downloads ~10-12 GB of model weights — be patient at step 4.
# =============================================================================
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
SCRIPT = WORK / "z_image_turbo_test.py"
BRANCH = "arena/83149951-areanatestpiolot"
RAW_URL = f"https://raw.githubusercontent.com/pmanjit/AreanaTestPiolot/{BRANCH}/z_image_turbo_test.py"
# (loads the latest committed script on that branch on every run; to pin an
#  immutable version, replace {BRANCH} with a full 40-char commit SHA)

NUM_IMAGES = "4"   # <-- set "1" for the quickest smoke test


def run(cmd, check=True, **kw):
    """Run a command, streaming its output live into this cell.
    Returns the exit code; raises only when check=True (the default)."""
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=env, **kw) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {' '.join(str(c) for c in cmd)}")
    return p.returncode


# 1) GPU check -----------------------------------------------------------------
run(["nvidia-smi"])

# 2) Install: recent diffusers (ZImagePipeline) + SDNQ (T4 INT8 path) ---------
run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "git+https://github.com/huggingface/diffusers.git", "sdnq"])

# 2b) Fix torchao clash: diffusers>=0.41 imports `FqnToConfig`, which only
# exists in torchao>=0.15 — Kaggle bundles torchao 0.14.x. Upgrade torchao
# WITHOUT letting pip touch Kaggle's torch (--no-deps), then preflight the
# actual import. If that still fails, remove torchao entirely (our SDNQ
# INT8 path never uses it; diffusers then cleanly skips it) and re-check.
run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps",
     "torchao>=0.15.0"])

PREFLIGHT = textwrap.dedent("""
    import importlib.metadata as m
    for p in ("diffusers", "transformers", "accelerate", "torchao", "sdnq"):
        try:
            print(p, m.version(p))
        except Exception:
            print(p, "NOT INSTALLED")
    try:
        from torchao.quantization import FqnToConfig
        print("torchao FqnToConfig: OK")
    except Exception as e:
        print("torchao check:", e)
    from diffusers import ZImagePipeline
    print("PREFLIGHT OK: ZImagePipeline import succeeded")
""")


def preflight():
    return run([sys.executable, "-c", PREFLIGHT], check=False) == 0


if not preflight():
    print("\nPreflight failed -> removing torchao (unused by the SDNQ path) and retrying...")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    if not preflight():
        raise RuntimeError("Preflight still fails after torchao fix — please share the "
                           "full output above (versions + traceback).")

# 3) Download the committed test script from GitHub ----------------------------
try:
    urllib.request.urlretrieve(RAW_URL, SCRIPT)
    print(f"\nDownloaded {SCRIPT} ({SCRIPT.stat().st_size} bytes) from branch {BRANCH}")
except Exception as e:  # fallback: shallow git clone of the branch
    print(f"\nRaw download failed ({e}); falling back to git clone...")
    repo_dir = WORK / "AreanaTestPiolot"
    if not repo_dir.exists():
        run(["git", "clone", "--depth", "1", "-b", BRANCH,
             "https://github.com/pmanjit/AreanaTestPiolot.git", str(repo_dir)])
    SCRIPT = repo_dir / "z_image_turbo_test.py"

# 4) Run the test (fresh interpreter -> picks up the just-installed diffusers) -
run([sys.executable, str(SCRIPT), "--num-images", NUM_IMAGES])

# 5) Show the results ----------------------------------------------------------
from IPython.display import Image, display

out_dir = WORK / "z_image_turbo_test_outputs"
display(Image(filename=str(out_dir / "contact_sheet.png")))
print((out_dir / "run_summary.json").read_text()[:1500])
print("\nDone. Images + contact_sheet.png + run_summary.json are in:", out_dir)
