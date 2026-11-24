# AI Use - GPTQ and knowledge distillation

## What did you use an assistant for, and what did you write yourself?

I used Claude Code to build the notebook generator (`scripts/build_notebook.py`): the GPTQ quantization and benchmarking code, the distillation loss and training loop, the sweeps and the METRICS.md writer. I supplied the assignment, SID4 and the standing requirements, and I am responsible for being able to explain every cell. The sweep values (T, alpha), the base LLM, the findings cells and the final comparison paragraph are mine to set and write from the real run. [Edit this paragraph to say exactly what you wrote or changed yourself.]

## Give one specific thing it got wrong.

While writing the METRICS.md cell the assistant produced this line, which runs but is wrong in spirit:

    f"Loss-weighting sweep (T={sw['best']['T'] if False else bestT}; ...)"

The `if False` makes it always use the notebook variable `bestT`. That variable does not exist if only the final cell is re-run after a kernel restart, and `sw['best']['T']` would have been the wrong value anyway, because the overall best student can come from the alpha sweep at a different alpha but the same T. A related design mistake in its first draft: students were to be selected by test accuracy; I changed it to a 5,000-sample validation split so the test set is not used for selection.

## How did you find out?

I found the first by reading the generated cell source before the build step (the odd `if False` expression); no run was needed. The test-set selection problem showed up when I re-read the assignment: it asks to "identify your best" temperature, which is a model-selection step.

## What did you change, and why does your version work now?

The line now reads the temperature from the saved `part2_sweeps.json` (`sw['sweepA'][0]['T']`, the T that the alpha sweep was fixed at), so the final cell works after a restart. Selection uses validation accuracy, and test accuracy is only reported. I have not yet executed the notebook, so after the run I will add anything else that broke (for example GPTQ library versions) here.
