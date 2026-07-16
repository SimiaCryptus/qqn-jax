# QQN Architecture Diagrams

QQN is fundamentally a **software engineering solution** to continuous
optimization: it factors a monolithic "update rule" into four orthogonal,
independently swappable components (Gradient, Oracle, Search, Region) that
compose through a single immutable state object. This document uses Mermaid
diagrams to make that architecture legible.

Companion references:

- [`algorithm.md`](../../algorithm.md) — the comprehensive algorithm reference.
- [`oracles.md`](../../oracles.md) — the oracle abstraction.
- [`regions.md`](../../regions.md) — projective regions.
- [`spline_search.md`](../../spline_search.md) — the spline line search.
- [`equivalences.md`](../../equivalences.md) — classical-method equivalences.
- [`notation.md`](../../notation.md) — symbol reference.

---

## 1. The Four Axes as Orthogonal Components

The central software-engineering claim: QQN separates concerns into four
conceptually orthogonal axes, each a pluggable interface. This is the "ports and
adapters" pattern applied to optimization.

```mermaid
graph TB
    subgraph QQN["QQN Combiner"]
        direction TB
        G["<b>Gradient</b><br/>-∇f(x)<br/><i>anchors path tangent d'(0)</i>"]
        O["<b>Oracle</b><br/>-H∇f(x)<br/><i>supplies t=1 endpoint d(1)</i>"]
        S["<b>Search</b><br/>walk t ∈ [0,1]<br/><i>enforce sufficient decrease</i>"]
        R["<b>Region</b><br/>project_R(x, x+d(t))<br/><i>optional feasibility</i>"]
    end

    G -->|"defines path start"| PATH["Quadratic Path<br/>d(t) = t(1-t)(-∇f) + t²(-H∇f)"]
    O -->|"defines path end"| PATH
    PATH -->|"one-dimensional<br/>search space"| S
    S -->|"each candidate"| R
    R -->|"projected candidate"| EVAL["Evaluate f(x + d_R(t))"]
    EVAL -->|"accepted step"| G

    style QQN fill:#e8f0fe,stroke:#4285f4
    style PATH fill:#fef7e0,stroke:#fbbc04
    style G fill:#e6f4ea,stroke:#34a853
    style O fill:#fce8e6,stroke:#ea4335
    style S fill:#f3e8fd,stroke:#a142f4
    style R fill:#e0f7fa,stroke:#00acc1
```

**Why this matters (SW engineering):** each axis is a `NamedTuple` interface
(`Oracle`, `Region`) with `init` / `direction`-or-`project` / `update`
callbacks. Swapping an implementation never touches the others — the very
definition of loose coupling.

---

## 2. The Quadratic Path as the Reframing

"Which direction?" becomes "where on the curve?". The path is the key data
structure that turns a discrete choice into a continuous, searchable object.

```mermaid
graph LR
    X["Current iterate x<br/>d(0) = 0"]
    subgraph CURVE["Quadratic Path d(t)"]
        T0["t → 0<br/>d'(0) = -∇f<br/><i>steepest descent</i>"]
        TM["0 < t < 1<br/>smooth blend<br/><i>grad·t(1-t) + oracle·t²</i>"]
        T1["t = 1<br/>d(1) = -H∇f<br/><i>oracle endpoint</i>"]
    end

    X --> T0
    T0 --> TM
    TM --> T1

    T0 -.->|"robust<br/>(global convergence)"| ROB["✓ always descends<br/>for small t"]
    T1 -.->|"fast<br/>(superlinear)"| FAST["✓ quasi-Newton<br/>near optimum"]

    style CURVE fill:#fef7e0,stroke:#fbbc04
    style ROB fill:#e6f4ea,stroke:#34a853
    style FAST fill:#fce8e6,stroke:#ea4335
```

---

## 3. The Solver Loop (Control Flow)

A JAXopt-style `init_state` / `update` / `run` interface. The entire loop is a
`lax.while_loop` so it stays JIT/vmap/grad-compatible — no host-side control
flow.

```mermaid
flowchart TD
    START([run]) --> INIT["init_state:<br/>eval f, ∇f, aux<br/>oracle.init, region.init<br/>error = ‖∇f‖"]
    INIT --> CHECK{"¬done ∧<br/>iter < maxiter?"}
    CHECK -->|no| DONE([return QQNState])
    CHECK -->|yes| UPDATE

    subgraph UPDATE["update (single iteration)"]
        direction TB
        U1["1. Oracle: qn_dir = oracle.direction(...)<br/>→ t=1 endpoint -H∇f"]
        U2["2. Gradient: grad_dir = -∇f"]
        U3["3. Search: walk d(t) over t∈[0,1]<br/>eval projected states x+d_R(t)"]
        U4["4. Select: new_params, new_value,<br/>new_grad, step_size t"]
        U5["5. Oracle update: push (s,y) if ⟨y,s⟩ > ε"]
        U6["6. Region update: adapt radius via ρ=ared/pred"]
        U7["7. Convergence: error = ‖∇f_new‖<br/>done = error ≤ tol; iter++"]
        U1 --> U2 --> U3 --> U4 --> U5 --> U6 --> U7
    end

    UPDATE --> CHECK

    style UPDATE fill:#e8f0fe,stroke:#4285f4
    style INIT fill:#e6f4ea,stroke:#34a853
    style DONE fill:#fce8e6,stroke:#ea4335
```

---

## 4. State Threading (Data Flow)

All state lives in a single immutable `QQNState` NamedTuple. This is the
"single source of truth" pattern — no hidden mutable buffers.

```mermaid
graph TB
    subgraph STATE["QQNState (immutable, JIT-compatible)"]
        direction LR
        F1["iter"]
        F2["value f(x)"]
        F3["grad ∇f(x)"]
        F4["oracle_state<br/><i>L-BFGS history /<br/>momentum buffer</i>"]
        F5["step_size t"]
        F6["error ‖∇f‖"]
        F7["done"]
        F8["aux"]
        F9["region_state<br/><i>trust radius</i>"]
    end

    UPD["update(state) → state'"]
    STATE --> UPD
    UPD -->|"new immutable copy"| STATE2["QQNState'"]

    F4 -.->|"threaded through"| ORACLE["oracle.update(OracleInfo)"]
    F9 -.->|"threaded through"| REGION["region.update(RegionInfo)"]
    ORACLE --> UPD
    REGION --> UPD

    style STATE fill:#e8f0fe,stroke:#4285f4
    style STATE2 fill:#e8f0fe,stroke:#4285f4
```

---

## 5. The Oracle Interface & Implementations (Strategy Pattern)

The oracle is a classic Strategy: one interface, many interchangeable
implementations. Combinators (`Fallback`, `Blend`) are the Composite pattern.

```mermaid
classDiagram
    class Oracle {
        <<interface>>
        +init(params) OracleState
        +direction(params, grad, state) (Direction, OracleState)
        +update(state, info) OracleState
    }

    class LBFGSOracle {
        history_size = 10
        +two-loop recursion
    }
    class MomentumOracle {
        beta = 0.9
        +v = βv + (1-β)∇f
    }
    class SecantOracle {
        alpha0, alpha_max
        +BB1 scalar step
    }
    class AndersonOracle {
        window, beta, reg
        +least-squares mix
    }
    class ShampooOracle {
        block_size, update_freq
        +inverse-root precond.
    }
    class Fallback {
        +children: List~Oracle~
        +first valid via jnp.where
    }
    class Blend {
        +weights: List~float~
        +convex combination
    }

    Oracle <|.. LBFGSOracle
    Oracle <|.. MomentumOracle
    Oracle <|.. SecantOracle
    Oracle <|.. AndersonOracle
    Oracle <|.. ShampooOracle
    Oracle <|.. Fallback
    Oracle <|.. Blend
    Fallback o-- Oracle : composes
    Blend o-- Oracle : composes
```

---

## 6. The Region Interface & Projected Path

Regions are pure projections applied *inside* the line search. The search
navigates the **projected path** `d_R(t)`, keeping descent guarantees on the
feasible set.

```mermaid
flowchart LR
    subgraph SEARCH["Line Search (walking t)"]
        direction TB
        PROBE["candidate = x + d(t)"]
        PROJ["region.project(x, candidate, state)"]
        DR["d_R(t) = projected - x"]
        EV["evaluate f(x + d_R(t))"]
        PROBE --> PROJ --> DR --> EV
    end

    EV -->|"sufficient decrease?"| ACC{Armijo /<br/>Wolfe}
    ACC -->|yes| ACCEPT([accept t])
    ACC -->|no| SHRINK["shrink/adjust t"]
    SHRINK --> PROBE

    subgraph REGIONS["Region implementations"]
        B["BoxRegion<br/>clip(candidate, lo, hi)"]
        OR["OrthantRegion<br/>zero sign-flips (OWL-QN)"]
        TR["TrustRegion<br/>radial clip ‖step‖ ≤ Δ"]
        SEQ["Sequential<br/>R_k ∘ ... ∘ R_1"]
    end

    PROJ -.-> REGIONS

    style SEARCH fill:#e0f7fa,stroke:#00acc1
    style REGIONS fill:#f3e8fd,stroke:#a142f4
```

---

## 7. Line Search Strategies & the Spline Wrapper (Decorator Pattern)

The spline refinement is **not** a competing search but a Decorator that
*wraps* any inner search, reusing every probe as a control point.

```mermaid
flowchart TD
    subgraph INNER["Inner Line Search (Strategy)"]
        LS1["backtracking / armijo"]
        LS2["strong_wolfe (over-restricts)"]
        LS3["hager_zhang"]
        LS4["fixed"]
    end

    INNER -->|"spline=False"| RESULT1([LineSearchResult])

    INNER -->|"spline=True"| WRAP

    subgraph WRAP["spline_wrap(inner_search)"]
        direction TB
        W1["run inner search → accepted step (bv)"]
        W2["collect probes as control points<br/>(t_i, f_i, m_i)"]
        W3["reflect tangents vs secant<br/>if sign(m) ≠ sign(Δ)"]
        W4["fit cubic Hermite spline"]
        W5["solve f'(s)=0 → candidate steps"]
        W6{"improves = cf < bv?"}
        W1 --> W2 --> W3 --> W4 --> W5 --> W6
        W6 -->|yes| W7["adopt spline candidate"]
        W6 -->|no| W8["keep inner result"]
    end

    WRAP --> RESULT2([LineSearchResult])

    style WRAP fill:#fef7e0,stroke:#fbbc04
    style INNER fill:#f3e8fd,stroke:#a142f4
```

---

## 8. Spline Search Internals

Each probe carries both a fitness value and a slope. The cubic Hermite spline
honors both, and stationary points are found in closed form.

```mermaid
flowchart TD
    PROBE["Probe at t_i yields:<br/>f_i = f(d(t_i))<br/>m_i = ⟨∇f, d'(t_i)⟩"]
    PROBE --> STORE["Store control point<br/>(t_i, f_i, m_i), kept sorted by t"]
    STORE --> BRACKET["Active bracket:<br/>adjacent anchors (t_lo, t_hi)"]

    BRACKET --> SECANT["Secant slope<br/>Δ = (f_1 - f_0) / h"]
    SECANT --> REFLECT{"sign(m) ≠ sign(Δ)<br/>and Δ ≠ 0?"}
    REFLECT -->|yes| FLIP["m ← -m<br/><i>upstream/downstream symmetry</i>"]
    REFLECT -->|no| KEEP["keep raw m<br/><i>(flat secant: essential)</i>"]

    FLIP --> HERMITE
    KEEP --> HERMITE

    HERMITE["Cubic Hermite segment:<br/>f(s) = h00·f_0 + h10·h·m_0<br/>+ h01·f_1 + h11·h·m_1"]
    HERMITE --> ROOTS["Solve f'(s) = As² + Bs + C = 0<br/>quadratic formula, guard |A|<ε"]
    ROOTS --> CAND["Candidates with s ∈ [0,1]<br/>→ t = t_0 + s·h"]
    CAND --> BEST["Select lowest predicted fitness"]

    style REFLECT fill:#fce8e6,stroke:#ea4335
    style HERMITE fill:#fef7e0,stroke:#fbbc04
    style BEST fill:#e6f4ea,stroke:#34a853
```

---

## 9. Equivalences: Configuration Space

Classical methods are just points in QQN's configuration space where one or two
axes are fixed. This is the "one framework, many products" payoff.

```mermaid
graph TB
    subgraph CONFIG["QQN Configuration Space"]
        direction TB
        AXES["(Oracle, Search, Region)"]
    end

    AXES --> GD["Gradient Descent<br/>any oracle, fixed/t→0, None"]
    AXES --> LBFGS["L-BFGS<br/>lbfgs, reach t=1, None"]
    AXES --> NEWTON["Newton<br/>exact-Hessian, accept t=1, None"]
    AXES --> MOM["Momentum / Heavy-Ball<br/>momentum, fixed t=1, None"]
    AXES --> BB["Barzilai-Borwein<br/>secant, accept t=1, None"]
    AXES --> AND["Anderson Acceleration<br/>anderson, accept t=1, None"]
    AXES --> TR["Trust Region<br/>lbfgs, ρ-acceptance, TrustRegion"]
    AXES --> OWL["OWL-QN<br/>lbfgs, line search, OrthantRegion"]
    AXES --> PG["Projected Gradient<br/>any, t→0, BoxRegion"]

    style CONFIG fill:#e8f0fe,stroke:#4285f4
    style GD fill:#e6f4ea,stroke:#34a853
    style LBFGS fill:#e6f4ea,stroke:#34a853
    style TR fill:#e0f7fa,stroke:#00acc1
    style OWL fill:#e0f7fa,stroke:#00acc1
    style PG fill:#e0f7fa,stroke:#00acc1
    style MOM fill:#fce8e6,stroke:#ea4335
    style BB fill:#fce8e6,stroke:#ea4335
    style AND fill:#fce8e6,stroke:#ea4335
```

---

## 10. The Robustness/Speed Trade-off QQN Resolves

The historical tension and how the path's endpoints dissolve it.

```mermaid
quadrantChart
    title Robustness vs Speed
    x-axis "Slow" --> "Fast"
    y-axis "Fragile" --> "Robust"
    quadrant-1 "Ideal"
    quadrant-2 "Safe but slow"
    quadrant-3 "Worst"
    quadrant-4 "Risky"
    "Gradient Descent": [0.2, 0.9]
    "Momentum": [0.4, 0.75]
    "L-BFGS": [0.85, 0.3]
    "Newton": [0.95, 0.2]
    "QQN": [0.8, 0.85]
```

QQN lands in the "Ideal" quadrant because the path *begins* at the robust
gradient (`d'(0) = -∇f`) and *ends* at the fast oracle (`d(1) = -H∇f`); the line
search picks the best point in between, inheriting global convergence from the
tangent and superlinear speed from the endpoint.

---

## 11. Compilation & Transform Composition (Why JAX)

Pure functional design is what keeps the four axes genuinely modular while
remaining fully traceable.

```mermaid
graph LR
    SRC["Pure functional QQN<br/>(no host control flow)"]
    SRC --> JIT["jit<br/><i>compile the whole run</i>"]
    SRC --> VMAP["vmap<br/><i>batched start points</i>"]
    SRC --> PMAP["pmap<br/><i>multi-device</i>"]
    SRC --> GRAD["grad<br/><i>differentiate through run<br/>(meta-learning)</i>"]

    SRC -.->|"branching"| WHERE["jnp.where / lax.select"]
    SRC -.->|"loops"| SCAN["lax.while_loop / lax.scan"]

    style SRC fill:#e8f0fe,stroke:#4285f4
    style JIT fill:#e6f4ea,stroke:#34a853
    style GRAD fill:#fce8e6,stroke:#ea4335
```

---

## 12. Sequence: One Accepted Step (End-to-End)

A sequence diagram tying all components together for a single iteration.

```mermaid
sequenceDiagram
    participant Solver
    participant Oracle
    participant Search
    participant Region
    participant f as Objective f

    Solver->>Oracle: direction(params, grad, oracle_state)
    Oracle-->>Solver: qn_dir (-H∇f), state'
    Note over Solver: grad_dir = -∇f<br/>build d(t) = t(1-t)grad_dir + t²qn_dir

    loop walk t ∈ [0,1]
        Solver->>Search: propose t
        Search->>Region: project(x, x+d(t))
        Region-->>Search: projected candidate
        Search->>f: evaluate f(x + d_R(t)), ∇f
        f-->>Search: value, grad
        Search->>Search: check sufficient decrease
    end

    Search-->>Solver: accepted t, new_params, new_value, new_grad
    Solver->>Oracle: update(state, OracleInfo(s, y, t, α))
    Oracle-->>Solver: oracle_state''
    Solver->>Region: update(region_state, RegionInfo(ared, pred, t))
    Region-->>Solver: region_state''
    Note over Solver: error = ‖∇f_new‖<br/>done = error ≤ tol
```

---

## Diagram Index

| # | Diagram                        | Teaches                                  |
|---|--------------------------------|------------------------------------------|
| 1 | Four axes                      | Separation of concerns / ports & adapters |
| 2 | Quadratic path                 | The core reframing (direction → curve)   |
| 3 | Solver loop                    | Control flow / `lax.while_loop`          |
| 4 | State threading                | Immutable single source of truth         |
| 5 | Oracle interface               | Strategy + Composite patterns            |
| 6 | Region / projected path        | Projection inside the search             |
| 7 | Spline wrapper                 | Decorator pattern                        |
| 8 | Spline internals               | Hermite interpolation + reflection       |
| 9 | Equivalences                   | Configuration space → classical methods  |
|10 | Trade-off quadrant             | Why QQN blends rather than chooses       |
|11 | Transform composition          | Why pure functional JAX                  |
|12 | Sequence of one step           | End-to-end component interaction         |