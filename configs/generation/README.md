# Generation configs

These configs apply the paper's default generation settings to the five downstream ReID datasets: Tiger (ATRW), Nyala, CowDataset, IPanda50, and Stoat. All five use the shared WildIcon generation model and parameter settings, organized by dataset. Each reference produces two 81-frame videos at 832 × 480 and 16 FPS, using 50 sampling steps. The adapter uses a frozen DINOv3 ViT-L/16 encoder and locally prepared foreground images.

Pair a trained WildIcon adapter checkpoint with a config to generate videos using `tools/generate.py`; see the [inference command](../../README.md#inference). The [training instructions](../../README.md#training) describe how to train the adapter. Obtain Wan and DINOv3 weights from their original sources as described in [Installation](../../README.md#installation).

For downstream ReID augmentation, select references only from the real training split. Prepare two mild motion prompts per reference, preserving the visible side and markings. Exclude full rotations, opposite-side views, and camera orbits. Check candidate videos for unsupported identity-bearing regions before assigning source identities or applying the paper's frame-similarity filter. Query and gallery remain real-only.

The runner checks reference and foreground files before loading models. `--dry-run` validates the inputs and prints the resolved command. `--resume` completes a partial run using existing videos; `--force` regenerates the run. Keep the checkpoint and inputs unchanged when resuming.

Each completed run stores `generation_manifest.csv`, `generation_config.json` (resolved arguments, prompts, and checkpoint SHA-256), and `animal_config.json` next to `generated_videos/`.
