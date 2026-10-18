# =============================================================================
# Z-Image-Turbo stickman test — KAGGLE SINGLE CELL (paste this whole thing
# into ONE cell of a new Kaggle notebook and run it)
#
# Kaggle Settings first:  Accelerator = GPU T4 x2   |   Internet = ON
#
# What it does: GPU check -> installs diffusers(GitHub)+sdnq -> downloads
# z_image_turbo_test.py from the committed GitHub branch -> runs it
# (4 stickman images, 1024px) -> displays the contact sheet + timings.
# Everything runs in subprocesses, so no kernel restart is needed.
# First run downloads ~10-12 GB of model weights — be patient at step 4.
# =============================================================================
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
SCRIPT = WORK / "z_image_turbo_test.py"
BRANCH = "arena/83149951-areanatestpiolot"
RAW_URL = f"https://raw.githubusercontent.com/pmanjit/AreanaTestPiolot/{BRANCH}/z_image_turbo_test.py"
# Immutable alternative (exact commit 3d0e4afe...):
# RAW_URL = "https://raw.githubusercontent.com/pmanjit/AreanaTestPiolot/3d0e4afe439ac794574cc9b6a063ce5981a5f5e2/z_image_turbo_test.py"

NUM_IMAGES = "4"   # <-- set "1" for the quickest smoke test


def run(cmd, **kw):
    """Run a command, streaming its output live into this cell."""
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=env, **kw) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {' '.join(str(c) for c in cmd)}")


# 1) GPU check -----------------------------------------------------------------
run(["nvidia-smi"])

# 2) Install: recent diffusers (ZImagePipeline) + SDNQ (T4 INT8 path) ---------
run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "git+https://github.com/huggingface/diffusers.git", "sdnq"])

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
