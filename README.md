# Text2Noise

Research code for two related audio tasks:

1. **Text-to-Noise** — train conditional score models and generate noise from class labels or free-form captions.
2. **Speech Enhancement** — separate speech and noise from a noisy mixture using a speech prior together with one of the trained noise priors.

The repository keeps the shared model code in one place and exposes each experimental version through a dedicated Git branch. Model checkpoints, datasets, generated results, and account credentials are intentionally not included.

## Version branches

| Version | Conditioning / method | Main entrypoints | Branch |
| --- | --- | --- | --- |
| v1 (default) | Stored text embeddings | `run_conditioned_pipeline.sh`, `inference.py`, `eval/evaluation.py` | `main` |
| CoNeTTE | Descriptive CoNeTTE captions | `run_conditioned_pipeline_conette.sh`, `inference_conette.py`, `eval/evaluation_conette.py` | `versions/conette` |

There are exactly two maintained branches. `main` is the v1 implementation, and `versions/conette` contains the CoNeTTE-conditioned implementation. Version-specific launchers, training modules, data modules, and evaluation entrypoints exist only on their corresponding branch. The `demo/` directory and the speech-enhancement notebook are preserved on both branches.

## Repository layout

```text
.
├── inference*.py              # Text-to-Noise inference entrypoints
├── train.py                   # Score-model training
├── run_*pipeline*.sh          # Text-to-Noise launchers
├── eval/                      # Speech-enhancement evaluation and launchers
├── sgmse/                     # Data, SDE, score-model, backbone, and AV utilities
├── src/                       # Speech/noise inference algorithms and metrics
├── demo/                      # Small generated audio examples
├── demo.ipynb                 # Speech-enhancement demo
└── requirements.txt
```

Generated metadata, checkpoints, logs, cluster output, caches, and evaluation results are excluded by `.gitignore`.

## Setup

Python 3.10 is recommended. Create an isolated environment and install the captured research dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA and PyTorch must match the GPU and driver available on your machine. The dependency file was exported from the original research environment, so a fresh platform may require compatible PyTorch/CUDA package selections.

## Configuration

Copy the example environment file, replace the placeholders, and load it before running a launcher:

```bash
cp .env.example .env
# Edit .env with local paths. Never commit it.
source .env
```

The example file documents the values required by each launcher. Keep all machine-specific locations and credentials in the untracked `.env` file. Checkpoints and datasets remain external because they can be large and may have separate licenses.

## Demos

There are two separate demos:

1. **Noise generation:** use the inference script provided by the selected branch. On `main`, run `inference.py`; on `versions/conette`, run `inference_conette.py`.
2. **Speech enhancement:** use `demo.ipynb`. The notebook combines a speech-prior checkpoint with a branch-compatible noise checkpoint to separate a noisy mixture into speech and noise.

The following example runs the v1 noise-generation demo without placing local paths in the repository:

```bash
PATH_TEST="<metadata-file>"
PATH_CHECKPOINT="<noise-checkpoint-file>"
PATH_OUTPUT="<output-directory>"

python inference.py \
  --prompt "<text-prompt>" \
  --metadata-jsonl "$PATH_TEST" \
  --diffusion-checkpoint "$PATH_CHECKPOINT" \
  --output-dir "$PATH_OUTPUT"
```

For speech enhancement, start Jupyter, open `demo.ipynb`, and set its neutral configuration values to files on your machine:

```bash
jupyter notebook demo.ipynb
```

## Task 1: Text-to-Noise

The default `main` branch is v1:

```bash
git switch main
bash run_conditioned_pipeline.sh
```

For generation, use the noise-generation demo above. The stored-embedding model requires metadata created for the same checkpoint. The CoNeTTE model instead accepts a descriptive caption.

## Task 2: Speech Enhancement

Speech enhancement combines an external speech-prior checkpoint with a Text-to-Noise checkpoint. Configure the required local values in `.env` and load it before running a launcher.

Run v1 locally in ten segments:

```bash
git switch main
TOTAL_SEGMENTS=10 bash run_eval_activate.sh
```

Run CoNeTTE-based speech enhancement from its model branch:

```bash
git switch versions/conette
TOTAL_SEGMENTS=10 bash run_eval_activate_conette.sh
```

The `eval/launch_SE_ALL*.sh` and `eval/single_seg_launch_SE*.sh` scripts are optional Grid'5000/OAR launchers. The Python evaluation entrypoints can also be called directly; use `python eval/evaluation.py --help` or `python eval/evaluation_conette.py --help` for their full arguments.

## Checkpoints and data

This repository does not distribute training datasets or noise-model checkpoints. Keep them outside Git and configure their locations with environment variables or command-line arguments. Generated artifacts and local configuration are ignored by Git.

For the speech-enhancement demo, download the [speech-model checkpoint](https://huggingface.co/jeaneudesAyilo/enudiffuse/blob/main/separate_wsjqut_speech_modeling.ckpt). That checkpoint comes from the [EnuDiffSE repository](https://github.com/jeaneudesAyilo/enudiffuse); please follow its setup, citation, and license information when using the model.

## Acknowledgments

This work was conducted at **Inria Nancy** within the **MULTISPEECH** team. I would like to extend my sincere gratitude to my supervisor, **Dr. Mostafa Sadeghi**, for his invaluable guidance, support, and mentorship throughout this project.

Special thanks to the open-source communities behind the baseline architectures and utilities integrated into this repository:
* **Score-based models:** Adapted from [SGMSE](https://github.com/sp-uhh/sgmse) and [enudiffuse](https://github.com/jeaneudesAyilo/enudiffuse).
* **Speech enhancement stack:** Builds upon the referenced DiffUSE/UDiffSE frameworks.
* **Audio-Visual components:** Utilizes components derived from [AV-HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/av_hubert) / fairseq.

---

## License

This project is released under the [MIT License](LICENSE). 

> **Note:** Upstream codebases, pretrained checkpoints, and datasets incorporated or referenced in this repository maintain their respective original licenses (e.g., Fairseq, SGMSE). Please check upstream repository headers before commercial redistribution.