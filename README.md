# Text2Noise

Research code for two related audio tasks:

1. **Text-to-Noise** — train conditional score or flow-matching models and generate noise from class labels, free-form captions, or CLAP audio embeddings.
2. **Speech Enhancement** — separate speech and noise from a noisy mixture using a speech prior together with one of the trained noise priors.

The repository keeps the shared model code in one place and exposes each experimental version through a dedicated Git branch. Model checkpoints, corpora, generated results, and account credentials are intentionally not included.

## Version branches

| Task | Version | Conditioning / method | Main entrypoints | Branch |
| --- | --- | --- | --- | --- |
| Text-to-Noise | v1 | Combined noise dataset | `run_cont_pipeline.sh`, `run_pipeline.sh` | `versions/text-to-noise-v1` |
| Text-to-Noise | v2 | RoBERTa/AudioLDM text embeddings | `run_conditioned_pipeline.sh`, `inference.py` | `versions/text-to-noise-v2` |
| Text-to-Noise | v3 | CoNeTTE captions | `run_conditioned_pipeline_conette.sh`, `inference_conette.py` | `versions/text-to-noise-v3-conette` |
| Text-to-Noise | v4 | Optimal-transport flow matching | `run_conditioned_pipeline_fm.sh`, `train_fm.py` | `versions/text-to-noise-v4-flow-matching` |
| Text-to-Noise | v5 | CLAP audio embeddings | `run_conditioned_pipeline_v5.sh`, `inference_v5.py` | `versions/text-to-noise-v5-clap` |
| Speech Enhancement | v2 | v2 text-conditioned noise prior | `eval/SE_eval.sh`, `eval/evaluation.py` | `versions/speech-enhancement-v2` |
| Speech Enhancement | v3 | v3 CoNeTTE noise prior | `eval/SE_eval_v3.sh`, `eval/evaluation_v3.py` | `versions/speech-enhancement-v3-conette` |

`main` contains the consolidated research workspace. A version branch adds a short `VERSION.md` guide for that exact experiment while retaining the shared implementation needed to run it.

## Repository layout

```text
.
├── inference*.py              # Text-to-Noise inference entrypoints
├── train.py / train_fm.py     # Score-model and flow-matching training
├── run_*pipeline*.sh          # Text-to-Noise launchers
├── eval/                      # Speech-enhancement evaluation and launchers
├── sgmse/                     # Data, SDE, score-model, backbone, and AV utilities
├── src/                       # Speech/noise inference algorithms and metrics
├── demo/                      # Small generated audio examples
├── demo.ipynb                 # End-to-end research demo
└── requirements.txt
```

Generated metadata, checkpoints, logs, cluster output, caches, and evaluation results are excluded by `.gitignore`.

## Setup

Python 3.10 is recommended. Create an isolated environment and install the captured research dependencies:

```bash
git clone https://github.com/namdenn/Text2Noise.git
cd Text2Noise
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

Important variables are:

- `CONDA_INIT` and `CONDA_ENV_PATH` for shell launchers that use Conda.
- `CORPUS_ROOT` for the Text-to-Noise corpus.
- `TEXT_ENCODER_CHECKPOINT` for v2/v3 text conditioning.
- `SPEECH_CKPT`, `NOISY_ROOT`, and `CLEAN_ROOT` for speech enhancement.
- `WANDB_API_KEY` for online experiment logging. Set it only in your shell or `.env`; never place it in a tracked script.

All scripts default to the repository directory for `PROJ_ROOT`. Checkpoints and datasets remain external because they can be large and may have separate licenses.

## Task 1: Text-to-Noise

Expected corpus layout for the class-conditioned versions:

```text
$CORPUS_ROOT/
├── babble/{train,val,test}/**/*.wav
├── car/{train,val,test}/**/*.wav
├── cafe/{train,val,test}/**/*.wav
├── street/{train,val,test}/**/*.wav
├── lr/{train,val,test}/**/*.wav
└── white/{train,val,test}/**/*.wav
```

Select the version branch and run its launcher. For example, v2:

```bash
git switch versions/text-to-noise-v2
bash run_conditioned_pipeline.sh
```

Generate audio with the v2 model using the exact encoded metadata used during training:

```bash
python inference.py \
  --prompt "This is street noise" \
  --metadata-jsonl sgmse/metadata_combination_encoded_audioldm_cpt/test.jsonl \
  --diffusion-checkpoint logs/v2_01_08_2026/last.ckpt \
  --output-dir outputs/text_to_noise_v2
```

For v3, use `inference_conette.py` with a descriptive caption. For v5, use `inference_v5.py --help` to provide a CLAP-conditioned checkpoint and prompt. The v4 branch currently provides the flow-matching training path; it does not introduce a separate inference entrypoint.

## Task 2: Speech Enhancement

Speech enhancement combines an external speech-prior checkpoint with a Text-to-Noise checkpoint. Configure the corpus and checkpoint locations first:

```bash
export SPEECH_CKPT=/path/to/speech_prior.ckpt
export NOISY_ROOT=/path/to/noisy_eval_corpus
export CLEAN_ROOT=/path/to/clean_eval_corpus
```

Run v2 locally in ten segments:

```bash
git switch versions/speech-enhancement-v2
TOTAL_SEGMENTS=10 bash run_eval_activate.sh
```

Run v3/CoNeTTE in the same way:

```bash
git switch versions/speech-enhancement-v3-conette
TOTAL_SEGMENTS=10 bash run_eval_activate_v3.sh
```

The `eval/launch_SE_ALL*.sh` and `eval/single_seg_launch_SE*.sh` scripts are optional Grid'5000/OAR launchers. The Python evaluation entrypoints can also be called directly; use `python eval/evaluation.py --help` or `python eval/evaluation_v3.py --help` for their full arguments.

## Checkpoints and data

This repository does not distribute training corpora or model checkpoints. Keep them outside Git and configure their paths with environment variables or command-line arguments. Common local destinations such as `checkpoints/`, `data/`, `logs/`, `outputs/`, and generated metadata folders are ignored.

## Acknowledgments

The score-model implementation is adapted from [SGMSE](https://github.com/sp-uhh/sgmse). Parts of the speech-enhancement stack build on the referenced DiffUSE/UDiffSE research code, and the audio-visual utilities include AV-HuBERT/fairseq-derived components. See the source-file headers for method-specific papers and upstream attributions.

Developed for research within the MULTISPEECH team at Inria Nancy.

## License

No project-level license has been selected yet. Add a license before inviting reuse or redistribution, and verify the licenses of upstream code, datasets, checkpoints, and model weights separately.
