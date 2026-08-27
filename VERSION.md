# CoNeTTE Version

This branch uses descriptive CoNeTTE captions to condition the noise score model.

- Text-to-Noise training: `run_conditioned_pipeline_conette.sh`
- Text-to-Noise inference: `inference_conette.py`
- Caption encoding: `sgmse/data_module_conette.py`
- Speech Enhancement: `run_eval_activate_conette.sh` and `eval/evaluation_conette.py`

Set `TEXT_ENCODER_CHECKPOINT`, `METADATA_INPUT_DIR`, and `METADATA_OUTPUT_DIR` before training. Inference can use a free-form caption or an exact stored metadata embedding.
