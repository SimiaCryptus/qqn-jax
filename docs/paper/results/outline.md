Focused Report Tour:

* Reference optimizers vs best-of-breed
* Adam comparisons - Temperature and oracle
* Line search comparison
* Sine & Rolling-Sine case study - L-BFGS shows superios performance, compare and show PSD-region effect 

Results discussion notes

* L-BFGS and QQN both show an extra 2.6s on the first iteration due to jax compilation - need to explain and discuss
* LBFGS implementation only provides iterations - we don't have the eval count (need to implement an adaptive estimation workaround)
* Anomalous "Fix" strategy performance (open question for study)
