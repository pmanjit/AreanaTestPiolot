"""
Z-Image-Turbo — first test on Kaggle (2x T4): "a stickman sleeping in bed"
============================================================================

WHAT THIS DOES
--------------
Generates a few test images (1 raw prompt + engineered-prompt variations),
saves them + a contact-sheet grid + a JSON summary with per-image time and
peak VRAM. This tells us the three things we need before batch generation:
  1. Does the model actually run on a T4 without black-image / OOM issues?
  2. How many seconds per image?
  3. What does the quality / prompt-following look like?

WHY TWO MODEL PATHS?  (important T4 detail)
-------------------------------------------
Z-Image was built for BF16. Kaggle's T4 (Turing, sm_75) has NO native BF16
compute, and pure FP16 produces BLACK images (NaN latents) in this model
(see Tongyi-MAI/Z-Image GitHub issue #14). The proven T4 recipe from the
SDNQ author is: SDNQ INT8-quantized checkpoint + float32 compute dtype +
Triton quantized matmul (INT8 Tensor Cores), ~3x faster than FP32 and
~4x faster than emulated BF16 on a T4.

So this script auto-selects:
  * GPU sm_80+ (A100/L4/RTX 30xx/40xx ...): original  Tongyi-MAI/Z-Image-Turbo  in BF16
  * GPU sm_75  (T4 / RTX 20xx)             : quantized Disty0/Z-Image-Turbo-SDNQ-int8
                                             in FP32 + INT8 quantized matmul + CPU offload
Force a path any time with --model {auto,int8,uint4,original}.

NOTE: INT8 test quality ~= original quality (INT8 is near-lossless), but it
is still a quantized build. Judge final production quality accordingly.

KAGGLE SETUP (one time, ~3-5 min)
---------------------------------
1. Kaggle -> Code -> New Notebook.
2. Right panel -> Settings -> Accelerator -> "GPU T4 x2".  Internet -> ON.
   (This test uses ONE GPU only; the 2nd GPU matters later for batching.)
3. Add this file:  Add Input -> Upload -> upload z_image_turbo_test.py
   (or just import/upload the companion .ipynb notebook instead of this file)
4. Cell 1 (install — ZImagePipeline only exists in recent diffusers):
       !pip install -q --upgrade "git+https://github.com/huggingface/diffusers.git" sdnq
5. Cell 2 (run — adjust the path to where you uploaded the file):
       !python /kaggle/input/<your-dataset>/z_image_turbo_test.py
   First run downloads ~10-12 GB of weights. Images land in
   /kaggle/working/z_image_turbo_test_outputs/ (see Output tab).

EXAMPLES
--------
  python z_image_turbo_test.py                                  # default 4-image test
  python z_image_turbo_test.py --num-images 1                   # quickest smoke test
  python z_image_turbo_test.py --size 768                       # if you hit OOM
  python z_image_turbo_test.py --prompt "a cat astronaut" --num-images 3
  python z_image_turbo_test.py --model uint4                    # smallest/fastest, lower quality
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
RAW_PROMPT = "a stickman sleeping in bed"

# Z-Image-Turbo is photoreal-first. To actually get a *stickman*, the style
# has to be nailed down explicitly. There is no useful negative prompt
# (Turbo runs guidance_scale=0.0), so everything is phrased positively.
ENGINEERED_PROMPT = (
    "Minimalist black stick-figure line drawing on a plain white background, "
    "children's doodle style, clean thin black outlines only: one simple "
    "stickman sleeping peacefully in a bed, lying on his back, round head "
    "resting on a rectangular pillow, body hidden under a simple rectangular "
    "blanket, eyes drawn as two small closed curves, three handwritten letter "
    "'Z' floating above his head growing larger, a simple bedside table with "
    "a small lamp beside the bed, a crescent moon in a square window, "
    "lots of empty white space, flat 2D doodle, black ink on white paper"
)

MODEL_REPOS = {
    "original": "Tongyi-MAI/Z-Image-Turbo",
    "int8": "Disty0/Z-Image-Turbo-SDNQ-int8",
    "uint4": "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
}


def parse_args():
    p = argparse.ArgumentParser(description="Z-Image-Turbo Kaggle/T4 test")
    p.add_argument("--prompt", default=None,
                   help="Custom prompt for ALL images (default: stickman test set)")
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="Base seed; image i uses seed+i")
    p.add_argument("--size", type=int, default=1024, help="Square size px (multiple of 16)")
    p.add_argument("--steps", type=int, default=9,
                   help="9 steps = 8 DiT forwards (the official Turbo setting)")
    p.add_argument("--model", choices=["auto", "original", "int8", "uint4"], default="auto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer (slow 1st image; off by default)")
    p.add_argument("--no-raw-first", action="store_true",
                   help="Skip the raw-prompt comparison image")
    # parse_known_args so the script also survives being run inside Jupyter,
    # whose kernel injects its own argv (e.g. -f kernel.json).
    args, _unknown = p.parse_known_args()
    return args


def pick_model_mode(args, torch):
    if args.model != "auto":
        return args.model
    major, minor = torch.cuda.get_device_capability(args.gpu)
    # BF16 Tensor Cores exist from Ampere (sm_80) upwards. Below that (T4 etc.)
    # the SDNQ INT8 path is both faster and numerically safe.
    return "original" if major >= 8 else "int8"


def load_pipe(mode, args, torch):
    try:
        from diffusers import ZImagePipeline
    except ImportError as e:
        raise ImportError(
            "ZImagePipeline not found — your diffusers is too old. Run:\n"
            '  pip install --upgrade "git+https://github.com/huggingface/diffusers.git" sdnq\n'
            "then restart the Kaggle session and try again."
        ) from e

    repo = MODEL_REPOS[mode]
    device = f"cuda:{args.gpu}"

    if mode == "original":
        print(f"[load] {repo}  (BF16, full GPU)")
        pipe = ZImagePipeline.from_pretrained(
            repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        )
        pipe.to(device)
    else:
        print(f"[load] {repo}  (SDNQ quantized, FP32 compute + INT8 matmul + CPU offload)")
        # Importing SDNQConfig registers SDNQ into diffusers/transformers;
        # it MUST happen before from_pretrained() loads the quantized weights.
        from sdnq import SDNQConfig  # noqa: F401
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        # float32 on purpose: on T4, float16 => black images (NaN), and
        # bfloat16 breaks/slows Triton. Quantized INT8 matmul does the speed.
        pipe = ZImagePipeline.from_pretrained(repo, torch_dtype=torch.float32)

        if triton_is_available:
            pipe.transformer = apply_sdnq_options_to_model(
                pipe.transformer, use_quantized_matmul=True
            )
            pipe.text_encoder = apply_sdnq_options_to_model(
                pipe.text_encoder, use_quantized_matmul=True
            )
            print("[load] Triton quantized matmul: ON")
        else:
            print("[load] WARNING: Triton not available — dequant path will be SLOW.")

        # Full weights (~10 GB INT8 + FP32 leftovers) + activations don't love
        # a 16 GB T4; model CPU offload streams one component at a time.
        try:
            pipe.enable_model_cpu_offload(gpu_id=args.gpu)
        except TypeError:
            pipe.enable_model_cpu_offload()

    # Cheap VRAM insurance for the VAE decode at 1024px. Wrapped: older/newer
    # diffusers versions vary in which of these exist for DiT pipelines.
    for fn in ("enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass

    if args.compile:
        print("[load] torch.compile(transformer) — first image will take much longer")
        pipe.transformer = torch.compile(pipe.transformer)

    return pipe


def build_jobs(args):
    """Return list of (label, prompt, seed)."""
    if args.prompt:
        return [("custom", args.prompt, args.seed + i) for i in range(args.num_images)]
    jobs = []
    if not args.no_raw_first and args.num_images >= 1:
        jobs.append(("raw", RAW_PROMPT, args.seed))
    while len(jobs) < args.num_images:
        jobs.append(("engineered", ENGINEERED_PROMPT, args.seed + len(jobs)))
    return jobs


def image_stats(pil_image):
    import numpy as np

    a = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    return float(a.mean()), float(a.std())


def make_contact_sheet(paths, labels, out_path, thumb=512):
    from PIL import Image, ImageDraw

    cols = 2 if len(paths) > 1 else 1
    rows = (len(paths) + cols - 1) // cols
    footer = 28
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + footer)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, label) in enumerate(zip(paths, labels)):
        im = Image.open(p).convert("RGB").resize((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + footer)
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + thumb + 6), label, fill="black")
    sheet.save(out_path)
    return out_path


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("ERROR: No CUDA GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2.")

    if args.size % 16 != 0:
        args.size = max(256, (args.size // 16) * 16)
        print(f"[config] size adjusted to multiple of 16 -> {args.size}")

    gpu_name = torch.cuda.get_device_name(args.gpu)
    cc = torch.cuda.get_device_capability(args.gpu)
    total_gb = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
    print(f"[gpu] {gpu_name} | compute {cc[0]}.{cc[1]} | {total_gb:.1f} GB | "
          f"visible GPUs: {torch.cuda.device_count()} (using cuda:{args.gpu} only)")
    print(f"[torch] {torch.__version__} | cuda {torch.version.cuda}")

    mode = pick_model_mode(args, torch)
    print(f"[config] model mode = {mode}  ({MODEL_REPOS[mode]})")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("/kaggle/working/z_image_turbo_test_outputs")
        if Path("/kaggle/working").exists()
        else Path("z_image_turbo_test_outputs")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] output dir = {out_dir}")

    t0 = time.time()
    pipe = load_pipe(mode, args, torch)
    load_s = time.time() - t0
    print(f"[load] pipeline ready in {load_s:.1f}s")

    jobs = build_jobs(args)
    results, img_paths, sheet_labels = [], [], []

    for i, (label, prompt, seed) in enumerate(jobs):
        print(f"\n[gen {i + 1}/{len(jobs)}] label={label} seed={seed}")
        print(f"           prompt: {prompt[:110]}{'...' if len(prompt) > 110 else ''}")
        torch.cuda.reset_peak_memory_stats(args.gpu)
        torch.cuda.synchronize(args.gpu)
        t = time.time()
        image = pipe(
            prompt=prompt,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,   # 9 -> 8 DiT forwards
            guidance_scale=0.0,               # Turbo: ALWAYS 0.0
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(args.gpu)
        gen_s = time.time() - t
        peak_gb = torch.cuda.max_memory_allocated(args.gpu) / 1024**3

        path = out_dir / f"stickman_{i:02d}_{label}_seed{seed}.png"
        image.save(path)
        mean, std = image_stats(image)
        is_black = mean < 2.0 and std < 2.0
        flag = "  <-- BLACK IMAGE! dtype problem (never use fp16 for this model)" if is_black else ""
        print(f"[gen {i + 1}/{len(jobs)}] saved {path.name} | {gen_s:.1f}s | "
              f"peak VRAM {peak_gb:.2f} GB | px mean {mean:.1f} std {std:.1f}{flag}")

        results.append({
            "index": i, "label": label, "prompt": prompt, "seed": seed,
            "file": str(path), "seconds": round(gen_s, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "pixel_mean": round(mean, 2), "pixel_std": round(std, 2),
            "black_image_suspected": is_black,
        })
        img_paths.append(path)
        sheet_labels.append(f"{label} | seed {seed} | {gen_s:.0f}s")

    sheet_path = make_contact_sheet(img_paths, sheet_labels, out_dir / "contact_sheet.png")
    gen_times = [r["seconds"] for r in results]
    summary = {
        "gpu": gpu_name, "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch": torch.__version__, "model_mode": mode, "model_repo": MODEL_REPOS[mode],
        "size": args.size, "steps": args.steps, "guidance_scale": 0.0,
        "load_seconds": round(load_s, 1),
        "per_image_seconds": gen_times,
        "mean_seconds_after_first": (
            round(sum(gen_times[1:]) / len(gen_times[1:]), 2) if len(gen_times) > 1 else gen_times[0]
        ),
        "max_peak_vram_gb": max(r["peak_vram_gb"] for r in results),
        "contact_sheet": str(sheet_path),
        "images": results,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(f"model            : {mode}  ({MODEL_REPOS[mode]})")
    print(f"load time        : {load_s:.1f}s (first ever run also downloads weights)")
    print(f"per-image time   : {gen_times} s")
    if len(gen_times) > 1:
        print(f"steady-state     : ~{summary['mean_seconds_after_first']} s/image "
              f"(excl. first)  =>  ~{3600 / summary['mean_seconds_after_first']:.0f} img/hour/GPU")
    print(f"max peak VRAM    : {summary['max_peak_vram_gb']:.2f} GB")
    print(f"contact sheet    : {sheet_path}")
    print(f"full summary     : {out_dir / 'run_summary.json'}")
    if any(r["black_image_suspected"] for r in results):
        print("!! Black image detected. Do NOT switch to float16. Keep the SDNQ/float32 "
              "path and report the run_summary.json contents.")
    print("=========================================")

    # If executed inside Jupyter (e.g. via %run), show the sheet inline.
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import Image as IPyImage, display
            display(IPyImage(filename=str(sheet_path)))
    except Exception:
        pass


if __name__ == "__main__":
    main()
