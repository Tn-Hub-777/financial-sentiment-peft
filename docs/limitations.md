# Limitations and reporting guardrails

Financial PhraseBank is a small, high-agreement benchmark, and its all-agree subset is easier than realistic financial text. It may overlap semantically with FinGPT training data and may have appeared in Qwen pretraining. The pipeline removes normalized exact and fuzzy overlaps from SFT, but this cannot prove the absence of paraphrase leakage or pretraining contamination.

The default pipeline uses MTEB's script-free parquet conversion of the 2,264 all-agree examples. Its source partitions are joined and labelled `allagree`; therefore `75agree`, `66agree`, and `50agree` are not silently substituted. Add a separately verified script-free source for those robustness sets before reporting them.

The DPO dataset contains US personal-finance advice preferences, not sentiment preferences. A gain in preference loss is not evidence of better sentiment classification; report the alignment tax by comparing M2 validation/PhraseBank macro-F1 to M1. Stop and flag an M2 run if macro-F1 is more than one absolute point below M1.

QLoRA adapter results are not identical to full fine-tuning, and a merged 4-bit adapter is not produced by this reference pipeline. Report GPU type, library versions, seed, trainable-parameter count, adapter disk size, peak allocated CUDA memory, and elapsed time. Treat a result above 95% macro-F1 as a leakage investigation trigger, not success.
