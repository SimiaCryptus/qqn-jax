# Notation and Symbol Reference

This page collects the symbols used across the QQN theory documents. Where a
symbol is overloaded, both meanings are listed with their governing context.

## Core Path

| Symbol   | Meaning                                              | Context       |
|----------|------------------------------------------------------|---------------|
| `x`      | current iterate (parameters)                         | everywhere    |
| `f(x)`   | objective value                                      | everywhere    |
| `∇f(x)`  | gradient                                             | everywhere    |
| `∇²f(x)` | Hessian                                              | Newton oracle |
| `H`      | approximate **inverse** Hessian (e.g. from L-BFGS)   | oracle        |
| `d(t)`   | quadratic path `t(1-t)(-∇f) + t²(-H∇f)`              | everywhere    |
| `d'(0)`  | initial path tangent, equals `-∇f`                   | everywhere    |
| `d_R(t)` | projected (feasible) path `project_R(x, x+d(t)) - x` | regions       |
| `t`      | path parameter in `[0, 1]`                           | everywhere    |
| `α`      | step size recovered by/from the line search          | search        |

## L-BFGS Two-Loop

| Symbol                  | Meaning                                                    |
|-------------------------|------------------------------------------------------------|
| `s = Δx = x_new − x`    | iterate difference (secant)                                |
| `y = Δ∇f = ∇f_new − ∇f` | gradient difference                                        |
| `ρᵢ = 1 / ⟨yᵢ, sᵢ⟩`     | curvature-pair scalar (**not** the TR ratio)               |
| `γ = ⟨y, s⟩ / ⟨y, y⟩`   | rolling initial-Hessian scale, `H₀ = γI`                   |
| `m`                     | history size (number of curvature pairs)                   |
| `ε`                     | curvature-safeguard threshold, admit pair iff `⟨y, s⟩ > ε` |

## Trust Region

| Symbol                    | Meaning                                          |
|---------------------------|--------------------------------------------------|
| `Δ`                       | trust radius                                     |
| `ared = f(x) − f(x_new)`  | actual reduction (a *decrease*, > 0 on progress) |
| `pred(t) = -⟨∇f, d(t)⟩`   | predicted reduction (along-path model)           |
| `ρ = ared / (pred + eps)` | acceptance ratio (**not** the L-BFGS `ρᵢ`)       |

## Spline Search

| Symbol                        | Meaning                                       |
|-------------------------------|-----------------------------------------------|
| `(t_i, f_i, m_i)`             | control point: position, fitness, slope       |
| `m_i = ⟨∇f(d(t_i)), d'(t_i)⟩` | directional derivative (**not** history size) |
| `h = t_1 − t_0`               | segment width                                 |
| `s = (t − t_0)/h`             | normalized segment parameter in `[0, 1]`      |
| `Δ = (f_1 − f_0)/h`           | segment secant slope (**not** trust radius)   |

## Overloaded Symbols — Quick Disambiguation

- **`ρ`**: L-BFGS curvature scalar `1/⟨y,s⟩` *vs.* trust-region ratio
  `ared/pred`. Disambiguate by document.
- **`m`**: L-BFGS history size *vs.* spline slope `m_i`. The slope always
  carries a subscript.
- **`Δ`**: trust radius *vs.* spline secant slope. The secant always appears
  as `Δ = (f_1 − f_0)/h`.

## See Also

- [`algorithm.md`](algorithm.md)
- [`oracles.md`](oracles.md)
- [`regions.md`](regions.md)
- [`spline_search.md`](spline_search.md)