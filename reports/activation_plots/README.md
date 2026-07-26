# Activation Function Gallery

Visual + mathematical reference for every activation registered in
[`experiments/models/activations.py`](../../experiments/models/activations.py).

Two registries feed this gallery:

| Registry                 | Source                                                          | Kind                       |
|--------------------------|-----------------------------------------------------------------|----------------------------|
| `UNIVARIATE_ACTIVATIONS` | `experiments/models/activations.py` (+ `spline_activations.py`) | elementwise `f: R -> R`    |
| `_ROLLING_ACTIVATIONS`   | `experiments/models/rolling_window_activation.py`               | ring-coupled `g: R^w -> R` |

`ACTIVATIONS = UNIVARIATE_ACTIVATIONS | _ROLLING_ACTIVATIONS` is the single lookup table used by `resolve_activation` /
`parse_activation`.

---

## Selecting an activation

Activations are chosen through the `ACTIVATION` environment variable (`parse_activation`, default `tanh,gelu` for
fashion, `tanh` for the sparse benchmark):

```bash
# one activation for every hidden layer
ACTIVATION=gelu python -m experiments...

# mixed spec: cycled across hidden layers (layer0=tanh, layer1=gelu, layer2=tanh, ...)
ACTIVATION=tanh,gelu python -m experiments...

# rolling-window (ring) activation
ACTIVATION=rolling_sin_diff python -m experiments...
```

Unknown names print a warning and fall back to the default (`sigmoid` inside `resolve_activation`).

### Plot conventions

* **Univariate plots** (`*.png`) show `y = f(x)` over a symmetric window around the origin — enough range to expose the
  periodicity of the wave / sawtooth / spline families.
* **Rolling plots** (`rolling/*.png`) visualise the *base coupling*
  `g(x, y)` (or a slice of `g(a, b, c)` for the 3-wide window) rather than a 1-D curve, because a rolling activation is
  a function of a unit **and its circular neighbours**.
* Both sets are generated from the registries themselves (`ACTIVATIONS`, `_ROLLING_BASE_FUNCTIONS`), so a new entry in
  the code automatically yields a new plot — the docs and the implementation cannot drift apart.

---

# 1. Univariate activations

## 1.1 Classic / monotone

| Name       | Definition                                                                                                                                     | Range                    | Plot                                 |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------|
| `identity` | $f(x) = x$                                                                                                                                     | $(-\infty, \infty)$      | <img src="identity.png" width="220"> |
| `relu`     | $f(x) = \max(0, x)$                                                                                                                            | $[0, \infty)$            | <img src="relu.png" width="220">     |
| `sigmoid`  | $f(x) = \dfrac{1}{1 + e^{-x}}$                                                                                                                 | $(0, 1)$                 | <img src="sigmoid.png" width="220">  |
| `tanh`     | $f(x) = \tanh x = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$                                                                                          | $(-1, 1)$                | <img src="tanh.png" width="220">     |
| `gelu`     | $f(x) = \tfrac{1}{2}x\left(1 + \tanh\!\big[\sqrt{2/\pi}\,(x + 0.044715x^{3})\big]\right)$ (JAX `approximate=True`; exact form is $x\,\Phi(x)$) | $\approx(-0.17, \infty)$ | <img src="gelu.png" width="220">     |
| `swish`    | $f(x) = x\,\sigma(x)$ (SiLU)                                                                                                                   | $\approx(-0.28, \infty)$ | <img src="swish.png" width="220">    |
| `softplus` | $f(x) = \log(1 + e^{x})$                                                                                                                       | $(0, \infty)$            | <img src="softplus.png" width="220"> |
| `abs`      | $f(x) = \lvert x\rvert$                                                                                                                        | $[0, \infty)$            | <img src="abs.png" width="220">      |
| `logabs`   | $f(x) = \operatorname{sign}(x)\,\log(1 + \lvert x\rvert)$                                                                                      | $(-\infty, \infty)$      | <img src="logabs.png" width="220">   |

Notes:

* `logabs` is an odd, sign-preserving log compressor — unbounded but with logarithmically shrinking gradient, useful for
  heavy-tailed pre-activations.
* `identity` is included so a "no nonlinearity" control run is a one-word config change.

## 1.2 Periodic families

All periodic activations are written in terms of the normalised phase
$\phi (x) = \dfrac{\omega x}{2\pi}$ and the *wrap* operator
$w (\phi) = \phi - \lfloor \phi + \tfrac12 \rfloor \in [-\tfrac12, \tfrac12)$.

### Sine, $f (x) = \sin (\omega x)$, period $2\pi/\omega$, range $[-1, 1]$

| Name    | $\omega$ | Plot                              |
|---------|----------|-----------------------------------|
| `sine`  | 1        | <img src="sine.png" width="220">  |
| `sine2` | 2        | <img src="sine2.png" width="220"> |
| `sine8` | 8        | <img src="sine8.png" width="220"> |

### Sawtooth, $f (x) = 2\,w\!\big (\tfrac{\omega x}{2\pi}\big)$, range $[-1, 1)$

Piecewise-linear odd ramp with a discontinuous reset each period.

| Name        | $\omega$ | Plot                                  |
|-------------|----------|---------------------------------------|
| `sawtooth`  | 1        | <img src="sawtooth.png" width="220">  |
| `sawtooth2` | 2        | <img src="sawtooth2.png" width="220"> |
| `sawtooth8` | 8        | <img src="sawtooth8.png" width="220"> |

### Triangle, $f (x) = 2\left\lvert 2\,w\!\big (\tfrac{\omega x}{2\pi}\big)\right\rvert - 1$, range $[-1, 1]$

Continuous (Lipschitz) periodic zig-zag: the rectified sawtooth, i.e. a piecewise-linear stand-in for $-\cos$.

| Name        | $\omega$ | Plot                                  |
|-------------|----------|---------------------------------------|
| `triangle`  | 1        | <img src="triangle.png" width="220">  |
| `triangle2` | 2        | <img src="triangle2.png" width="220"> |
| `triangle8` | 8        | <img src="triangle8.png" width="220"> |

The $\omega \in \{1, 2, 8\}$ ladder exists to probe how much *spectral bias*
is removed by giving a unit an intrinsically high-frequency response.

## 1.3 Radial / wavelet family

| Name          | Definition                                                         | Notes                                                                                            | Plot                                    |
|---------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------|
| `gaussian`    | $f(x) = e^{-x^{2}}$                                                | RBF-style bump, range $(0, 1]$, even                                                             | <img src="gaussian.png" width="220">    |
| `mexican_hat` | $f(x) = \dfrac{2}{\sqrt{3}\,\pi^{1/4}}\,(1 - x^{2})\,e^{-x^{2}/2}$ | Ricker wavelet = $-\,\partial^2_x$ Gaussian; zero mean, even, two negative side lobes            | <img src="mexican_hat.png" width="220"> |
| `morlet`      | $f(x) = \cos(5x)\,e^{-x^{2}/2}$                                    | Real Morlet: carrier $\omega_0 = 5$ under a unit-variance Gaussian envelope; band-pass, even     | <img src="morlet.png" width="220">      |
| `chirplet`    | $f(x) = \cos\!\big(3x + \tfrac12 x^{2}\big)\,e^{-x^{2}/2}$         | Instantaneous frequency $\omega(x) = 3 + x$ (chirp rate 1); localised **and** frequency-sweeping | <img src="chirplet.png" width="220">    |

These four are compactly supported (in practice) in *both* space and frequency, so a hidden unit acts as a localised
feature detector rather than a half-space indicator.

## 1.4 Cubic Hermite spline presets

Defined in [`spline_activations.py`](../../experiments/models/spline_activations.py). Given
knots $x_0 < \dots < x_{n-1}$, values $y_i$, and tangents $m_i$, on each segment with $h = x_{i+1} - x_i$
and $t = \operatorname{clip}\!\big (\tfrac{x - x_i}{h}, 0, 1\big)$:

$$
f (x) = h_{00} (t)\,y_i + h_{10} (t)\,h\,m_i + h_{01} (t)\,y_{i+1} + h_{11} (t)\,h\,m_{i+1}
$$

$$
h_{00} = 2t^{3} - 3t^{2} + 1,\quad h_{10} = t^{3} - 2t^{2} + t,\quad h_{01} = -2t^{3} + 3t^{2},\quad h_{11} = t^{3} - t^{2}
$$

* **Periodic presets** wrap the input, $x \leftarrow x_0 + \big ((x - x_0) \bmod P\big)$, and append a virtual
  knot $(x_0 + P,\, y_0,\, m_0)$ so the curve is
  $C^1$ across the seam.
* **Non-periodic presets** clamp $t$ to $[0, 1]$, which makes the spline **extrapolate flat** (saturating
  at $y_0$ / $y_{n-1}$) outside the knot range — the tangent basis terms vanish at the endpoints.

| Name                  | Knots $x_i$                                       | Values $y_i$             | Tangents $m_i$          | Period | Plot                                            |
|-----------------------|---------------------------------------------------|--------------------------|-------------------------|--------|-------------------------------------------------|
| `hermite_wave`        | $[-\pi, -\tfrac{\pi}{2}, 0, \tfrac{\pi}{2}, \pi]$ | $[0, -1, 0, 1, 0]$       | $[-1, 0, 1, 0, -1]$     | $2\pi$ | <img src="hermite_wave.png" width="220">        |
| `hermite_double_wave` | $[-\pi, -\tfrac{\pi}{2}, 0, \tfrac{\pi}{2}, \pi]$ | $[-1, 0, 1, 0, -1]$      | $[0, 0, 0, 0, 0]$       | $2\pi$ | <img src="hermite_double_wave.png" width="220"> |
| `hermite_pulse`       | $[0, \tfrac{\pi}{2}, \pi, 1.1\pi, 2\pi]$          | $[0, 1, 1, 0, 0]$        | $[0, 0, 0, 0, 0]$       | $2\pi$ | <img src="hermite_pulse.png" width="220">       |
| `hermite_ramp`        | $[0,\; 0.95 \cdot 2\pi]$                          | $[-1, 1]$                | $[0.5, 0.5]$            | $2\pi$ | <img src="hermite_ramp.png" width="220">        |
| `hermite_step`        | $[-3, -1, 0, 1, 3]$                               | $[-1, -0.9, 0, 0.9, 1]$  | $[0, 0.3, 1.2, 0.3, 0]$ | —      | <img src="hermite_step.png" width="220">        |
| `hermite_sramp`       | $[-4, -2, 0, 2, 4]$                               | $[-1, -0.7, 0, 0.7, 1]$  | $[0, 0.2, 0.6, 0.2, 0]$ | —      | <img src="hermite_sramp.png" width="220">       |
| `hermite_valley`      | $[-3, -1.5, 0, 1.5, 3]$                           | $[0, -0.5, -1, -0.5, 0]$ | $[0, -0.8, 0, 0.8, 0]$  | —      | <img src="hermite_valley.png" width="220">      |

Reading the presets:

* `hermite_wave` — smooth single bump per period (a sine-like shape built from 4 cubic segments).
* `hermite_double_wave` — zero tangents at every knot give a flatter, two-peak-per-period ripple.
* `hermite_pulse` — asymmetric plateau on $[\tfrac{\pi}{2}, \pi]$ with a fast fall and a long dead zone: a duty-cycled
  gate.
* `hermite_ramp` — sawtooth-like monotone ramp with a smooth 5% reset window.
* `hermite_step` / `hermite_sramp` — tanh-like squashers with **explicitly controlled** central slope (1.2 vs 0.6) and
  hard saturation at $\pm 1$.
* `hermite_valley` — even, non-monotone well (an RBF-like detector with linear-ish flanks) that saturates to 0
  outside $[-3, 3]$.

---

# 2. Rolling-window (ring) activations

A rolling-window activation treats the pre-activation vector
$h \in \mathbb{R}^{N}$ of a layer as a **circular ring** and slides a width-$w$
window over it. With base function $g:\mathbb{R}^{w} \to \mathbb{R}$,

$$
y_i = g\big (h_i,\, h_{i+1},\, \dots,\, h_{i+w-1}\big), \qquad \text{indices mod } N
$$

so $N$ inputs produce $N$ outputs (layer width is preserved) with **no extra parameters**. Implementation:
`rolling_window` builds the shifted views with
`jnp.roll(h, -k, axis=-1)` for `k = 0..w-1` and calls `g` on them, i.e. it is a weight-free depthwise "convolution" with
a nonlinear kernel.

Registry names are prefixed: base name `sin_diff` -> `ACTIVATION=rolling_sin_diff`. Below, $x = h_i$ and $y = h_{i+1}$
(and $a, b, c$ for $w = 3$);
$\sigma$ denotes the logistic sigmoid. Tiny signed floors ($\varepsilon \approx 10^{-3}$, or $10^{-6}$/$10^{-12}$ where
noted) are added to denominators/roots/logs so gradients stay finite — they are omitted from the formulas for
readability but are present in the code.

## 2.1 General couplings

| Name (`rolling_*`) | $w$ | Definition                                                                                         | Properties                                                                        | Plot                                              |
|--------------------|-----|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------|
| `sin_diff`         | 2   | $g = \sin(x - y)$                                                                                  | **Default.** Smooth, bounded $[-1,1]$, periodic, antisymmetric, purely relational | <img src="rolling/sin_diff.png" width="220">      |
| `atan2_ramp`       | 3   | $g = \operatorname{atan2}(a + b,\; b + c)$                                                         | Bounded $(-\pi, \pi]$; middle element couples both components                     | <img src="rolling/atan2_ramp.png" width="220">    |
| `euclidean`        | 2   | $g = \sqrt{x^{2} + y^{2} + 10^{-12}}$                                                              | Symmetric, radial magnitude, unbounded above, $\ge 0$                             | <img src="rolling/euclidean.png" width="220">     |
| `tan_ratio`        | 2   | $g = \tanh\!\big(\tan(x/y)\big)$                                                                   | Ratio-sensitive oscillation squashed to $(-1,1)$; asymmetric                      | <img src="rolling/tan_ratio.png" width="220">     |
| `xlog_share`       | 2   | $g = \dfrac{x}{S}\log\!\Big(\dfrac{\lvert y\rvert}{S}\Big),\; S = \lvert x\rvert + \lvert y\rvert$ | "Information share" coupling; asymmetric, normalised                              | <img src="rolling/xlog_share.png" width="220">    |
| `harmonic`         | 2   | $g = \dfrac{2xy}{x + y}$                                                                           | Harmonic mean (regularised pole); symmetric, rewards agreement                    | <img src="rolling/harmonic.png" width="220">      |
| `parallel`         | 2   | $g = \big(\tfrac1x + \tfrac1y\big)^{-1} = \dfrac{xy}{x+y}$                                         | "Parallel resistors" = half the harmonic mean; symmetric                          | <img src="rolling/parallel.png" width="220">      |
| `sum_over_prod`    | 2   | $g = \dfrac{x + y}{xy} = \tfrac1x + \tfrac1y$                                                      | Symmetric, amplifies small-magnitude signals                                      | <img src="rolling/sum_over_prod.png" width="220"> |
| `ratio`            | 2   | $g = x / y$                                                                                        | Raw scale-relative coupling; unbounded, asymmetric                                | <img src="rolling/ratio.png" width="220">         |
| `softmin`          | 2   | $g = -\log\!\big(e^{-x} + e^{-y}\big)$ (stable log-sum-exp form)                                   | Smooth minimum -> "soft AND"; symmetric                                           | <img src="rolling/softmin.png" width="220">       |

## 2.2 Fuzzy-logic gates

Both neighbours are squashed to soft bits $a = \sigma (x)$, $b = \sigma (y)$ and combined probabilistically. All are
smooth, symmetric, and bounded in $(0, 1)$.

| Name (`rolling_*`) | Definition        | Fires when             | Plot                                         |
|--------------------|-------------------|------------------------|----------------------------------------------|
| `soft_and`         | $g = ab$          | both strongly positive | <img src="rolling/soft_and.png" width="220"> |
| `soft_or`          | $g = a + b - ab$  | at least one positive  | <img src="rolling/soft_or.png" width="220">  |
| `soft_xor`         | $g = a + b - 2ab$ | the two **disagree**   | <img src="rolling/soft_xor.png" width="220"> |

## 2.3 Symmetric couplings, $g (x, y) = g (y, x)$

Direction-free ring mixing with genuine 2-D structure (multiple independent terms stay alive).

| Name (`rolling_*`)     | Definition                                                                                    | Plot                                                     |
|------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `sym_mixed_poly`       | $g = x^{2}y + y^{2}x = xy(x + y)$                                                             | <img src="rolling/sym_mixed_poly.png" width="220">       |
| `sym_sig_sum`          | $g = \sigma(x) + \sigma(y)$                                                                   | <img src="rolling/sym_sig_sum.png" width="220">          |
| `sym_sig_prod`         | $g = \sigma(x)\,\sigma(y)$                                                                    | <img src="rolling/sym_sig_prod.png" width="220">         |
| `sym_sig_hybrid`       | $g = \sigma(x) + \sigma(y) + \sigma(x)\sigma(y)$                                              | <img src="rolling/sym_sig_hybrid.png" width="220">       |
| `sym_interaction_norm` | $g = \dfrac{xy}{1 + x^{2} + y^{2}}$                                                           | <img src="rolling/sym_interaction_norm.png" width="220"> |
| `sym_param_kernel`     | $g = \sigma\!\big(w_1(x + y) + w_2 xy + w_3(x^{2} + y^{2})\big)$, fixed $w_1 = w_2 = w_3 = 1$ | <img src="rolling/sym_param_kernel.png" width="220">     |

## 2.4 Antisymmetric couplings, $g (x, y) = -g (y, x)$

Inject an *orientation* into the ring (the layer output changes sign if the ring is traversed backwards); all vanish on
the diagonal $x = y$.

| Name (`rolling_*`) | Definition                             | Plot                                                |
|--------------------|----------------------------------------|-----------------------------------------------------|
| `anti_diff`        | $g = x - y$ (linear finite difference) | <img src="rolling/anti_diff.png" width="220">       |
| `anti_tanh_diff`   | $g = \tanh(x - y)$, bounded $(-1, 1)$  | <img src="rolling/anti_tanh_diff.png" width="220">  |
| `anti_mixed_poly`  | $g = x^{2}y - y^{2}x = xy(x - y)$      | <img src="rolling/anti_mixed_poly.png" width="220"> |
| `anti_gated`       | $g = (x - y)\,\sigma(x + y)$           | <img src="rolling/anti_gated.png" width="220">      |
| `anti_normalized`  | $g = \dfrac{x - y}{1 + x^{2} + y^{2}}$ | <img src="rolling/anti_normalized.png" width="220"> |

## 2.5 Asymmetric couplings (no constraint)

Directed "flow" flavours: neither symmetric nor antisymmetric.

| Name (`rolling_*`)      | Definition                                                                                            | Plot                                                      |
|-------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `asym_directional_gate` | $g = x\,\sigma(y)$ (neighbour gates the unit)                                                         | <img src="rolling/asym_directional_gate.png" width="220"> |
| `asym_forward_biased`   | $g = \sigma(x) + \sigma(y) + \alpha\,\sigma(x)\sigma(y)$, $\alpha = 2$                                | <img src="rolling/asym_forward_biased.png" width="220">   |
| `asym_source_sink`      | $g = \sigma(x) - \sigma(y)$                                                                           | <img src="rolling/asym_source_sink.png" width="220">      |
| `asym_tiny_mlp`         | $g = \sigma\!\big(w_1 x + w_2 y + w_3 xy + w_4 x^{2}\big)$, $(w_1, w_2, w_3, w_4) = (1, 0.5, 1, 0.5)$ | <img src="rolling/asym_tiny_mlp.png" width="220">         |

---

# 3. Quick index

**Univariate (29):**
`identity`, `relu`, `sigmoid`, `tanh`, `gelu`, `swish`, `softplus`, `abs`,
`logabs`, `sine`, `sine2`, `sine8`, `sawtooth`, `sawtooth2`, `sawtooth8`,
`triangle`, `triangle2`, `triangle8`, `gaussian`, `mexican_hat`, `morlet`,
`chirplet`, `hermite_wave`, `hermite_double_wave`, `hermite_pulse`,
`hermite_ramp`, `hermite_step`, `hermite_sramp`, `hermite_valley`

**Rolling (28, prefix `rolling_`):**
`sin_diff`, `atan2_ramp`, `euclidean`, `tan_ratio`, `xlog_share`, `harmonic`,
`softmin`, `sum_over_prod`, `parallel`, `ratio`, `soft_xor`, `soft_and`,
`soft_or`, `sym_mixed_poly`, `sym_sig_sum`, `sym_sig_prod`, `sym_sig_hybrid`,
`sym_interaction_norm`, `sym_param_kernel`, `anti_diff`, `anti_tanh_diff`,
`anti_mixed_poly`, `anti_gated`, `anti_normalized`,
`asym_directional_gate`, `asym_forward_biased`, `asym_source_sink`,
`asym_tiny_mlp`

---

# 4. Adding a new activation

1. **Univariate:** add `"name": lambda x: ...` to `UNIVARIATE_ACTIVATIONS`
   in `experiments/models/activations.py`, or add a knot tuple
   `(xs, ys, ms, period)` to `_build_hermite_presets()` for a spline.
2. **Rolling:** add `def _my_fn(x, y): ...` in
   `experiments/models/rolling_window_activation.py` and register it in
   `_ROLLING_BASE_FUNCTIONS` as `"my_fn": (_my_fn, 2)`. The
   `rolling_my_fn` entry in `ACTIVATIONS` is derived automatically.
3. Keep denominators / logs / square roots guarded with signed
   $\varepsilon$ floors (see `_ratio`, `_parallel`, `_harmonic`) so the forward pass **and** its gradient stay finite.
4. Re-run the plotting utility; it iterates the registries, so the new entry appears here (`*.png` for univariate,
   `rolling/*.png` for ring couplings) with no further bookkeeping.