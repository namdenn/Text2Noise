# v2 — Default Version

`main` is the v2 branch. It uses stored 512-dimensional RoBERTa/AudioLDM-style text embeddings to condition the noise score model.

- Text-to-Noise training: `run_conditioned_pipeline.sh`
- Text-to-Noise inference: `inference.py`
- Data preparation: `sgmse/data_module.py`
- Speech Enhancement: `run_eval_activate.sh` and `eval/evaluation.py`

Set `CORPUS_ROOT` and `TEXT_ENCODER_CHECKPOINT` before training. Inference must receive the encoded JSONL used by the selected checkpoint so it loads the exact training condition.
