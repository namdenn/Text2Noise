# CoNeTTE Version

This branch uses descriptive CoNeTTE captions to condition the noise score model.

- Text-to-Noise training: `run_conditioned_pipeline_conette.sh`
- Text-to-Noise inference: `inference_conette.py`
- Caption encoding: `sgmse/data_module_conette.py`
- Speech Enhancement: `run_eval_activate_v3.sh` and `eval/evaluation_v3.py`

Set `TEXT_ENCODER_CHECKPOINT`, `CONETTE_METADATA_DIR`, and `CONETTE_ENCODED_DIR` before training. Inference can use a free-form caption or an exact stored metadata embedding.
