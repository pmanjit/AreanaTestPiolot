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

**Env note (Kaggle):** diffusers>=0.41 imports `FqnToConfig`, which needs
`torchao>=0.15.0`, but Kaggle bundles torchao 0.14.x. Handled three ways:
the installers upgrade torchao (`--no-deps`, so Kaggle's torch is untouched),
the single-cell loader preflights the import and removes torchao if needed
(the SDNQ path never uses it), and `z_image_turbo_test.py` shims
`ModuleFqnToConfig -> FqnToConfig` for torchao<0.15 before importing diffusers.
Outputs (`contact_sheet.png`, `run_summary.json` with seconds/image + peak VRAM)
are written to `/kaggle/working/z_image_turbo_test_outputs/`.
