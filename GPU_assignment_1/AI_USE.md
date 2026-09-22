# AI-Use Appendix - Sivasurya Chandran

**Not done yet** - this only covers the scaffolding stage. I still need to actually rent the GPU,
run the notebook, and then come back and fill in sections 2-4 with something real that happened
during that run.

## 1. Which parts did you use an assistant for, and which did you write yourself?

I had used AI to write the first draft of `GPU_Assignment_1.ipynb` and the script that generates it
(`scripts/build_gpu_assignment_notebook.py`), based on the assignment spec: the Step 0 GPU checks,
the Part B matmul sweep across precisions and sizes, the Part C bandwidth-vs-compute roofline math,
the Part D naive-vs-fused attention comparison with the OOM boundary search and the quadratic
memory fit, the Part E sustained-load thermal logger, and the Part F script that writes
`METRICS.md`. None of this has actually run on a real GPU yet, so none of the numbers or plots
exist - they only get generated once I rent a pod and actually execute it.

Before I turn this in I still need to: rent the RTX 4090/5090 pod, run the notebook start to
finish, sanity-check that the numbers make sense (BF16 should clearly beat FP32, the naive
attention memory curve should bend up way faster than the fused one, etc.), fill in the
reservation/GPU-hour table, and replace sections 2-4 below with an actual example of something
that went wrong during my run.

## 2. Give one specific thing it produced that was wrong. Paste the wrong output.

_(Filling this in after I actually run it - probably something like a wrong OOM boundary on the
first pass, the card-spec dictionary not matching whatever GPU I actually got, or an axis on one
of the plots being off.)_

## 3. How did you find out? What did the failure look like?

_(Filling this in after the run.)_

## 4. What did you change, and why does your version work?

_(Filling this in after the run.)_
