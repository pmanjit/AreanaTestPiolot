# AreanaTestPiolot
- [x] Start
- [ ] Succeed?
- [ ] Failed?

## Z-Image-Turbo Kaggle test
First test of Z-Image-Turbo on Kaggle T4 GPUs ("a stickman sleeping in bed"),
before building the batch-generation + vision-QC pipeline.

- `z_image_turbo_kaggle_test.ipynb` — **easiest**: upload/import into Kaggle
  (Create → Import Notebook), set Accelerator = GPU T4 x2, Internet = ON, Run All.
- `z_image_turbo_test.py` — same test as a standalone script
  (`python z_image_turbo_test.py --help` for options).

On T4 it auto-uses the SDNQ-INT8 build (FP16 gives black images in Z-Image;
BF16 has no native T4 support). On Ampere+ GPUs it uses the original BF16 model.
Outputs (`contact_sheet.png`, `run_summary.json` with seconds/image + peak VRAM)
are written to `/kaggle/working/z_image_turbo_test_outputs/`.
