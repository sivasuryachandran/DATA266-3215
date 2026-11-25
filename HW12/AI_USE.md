# AI Use - GPTQ and knowledge distillation

## What did you use an assistant for, and what did you write yourself?

I used Claude Code to write the notebook generator (`scripts/build_notebook.py`): the GPTQ quantization and benchmarking code, the distillation loss and training loop, the sweeps, and the METRICS writer. I supplied the assignment, my parameters and the standing requirements, set up the Colab runtime, uploaded the teacher, ran the notebook, and passed each error back to the assistant. The assistant also drafted `REPORT.md` and `README.md` from the saved outputs; I checked the numbers in them against `METRICS.md`. [Edit this paragraph so it says exactly what you wrote and changed yourself.]

## Give one specific thing it got wrong.

The GPTQ cell it wrote crashed on the first real run:

    ERROR AcceleratorError CUDA error: device-side assert triggered   (in optimum/gptq/quantizer.py, quantize_model)

It built calibration text by decoding 2048-token Wikitext windows and passing the strings to `GPTQConfig(model_seqlen=2048)`. Re-tokenizing adds a BOS token, so the sequences were 2049+ tokens, and OPT-125m only has 2048 positions. The same run also showed two other plausible-but-wrong pieces of its code: `load_dataset("wikitext", ...)` (rejected by the new huggingface_hub, which wants `Salesforce/wikitext`) and `ag["train"]["text"]` passed straight to the tokenizer (`ValueError: text input must be of type str`, because the new `datasets` returns a lazy Column, not a list). Its own first METRICS-cell draft also contained `{sw['best']['T'] if False else bestT}`, which would break after a kernel restart.

A measurement it got wrong without any error: in the first version the student's "peak memory" included the teacher's weights, because the teacher was still on the GPU; the table shows 1.70 GB for the student vs 1.43 GB for the teacher, which looks like the student uses more memory. I noticed this when writing the analysis (a smaller model cannot need more memory), and documented it in `METRICS.md` instead of re-running.

## How did you find out?

The CUDA assert and the Python tracebacks came from the Colab run (the cell outputs are saved in the notebook). The memory artifact I found by checking the table against the model sizes (67M vs 109M parameters).

## What did you change, and why does your version work now?

Calibration windows are 1024 tokens (`CALIB = 1024`, `model_seqlen=CALIB`), which stays well within OPT's 2048 positions even after a BOS is added; perplexity evaluation still uses 2048-token windows because it feeds token ids directly. I used the namespaced dataset ids (`Salesforce/wikitext`, `fancyzhx/ag_news`), wrapped the AG News columns in `list(...)`, removed the mismatched torchaudio/torchvision, and read the METRICS temperature from the saved sweep file. I did not fix the memory artifact, because that would need a GPU re-run; it is flagged in `METRICS.md`.
