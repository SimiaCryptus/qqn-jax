/**
     * TypeScript schema for the JSON artifacts written by
     * experiments/reporting/json_export.py.
     *
     * Two kinds of artifact exist:
     *   - "optimizer_run": one optimizer variant, written incrementally as
     *     each run completes (write_run_json).
     *   - "experiment": the aggregate record for a whole experiment
     *     (write_results_json).
     *
     * Schema version: 1.0.0
     */

    export type SchemaVersion = string;

    /** Description of the dataset the run was trained on. */
    export interface DatasetDescriptor {
      name: string | null;
      n_train: number | null;
      n_test: number | null;
      n_classes: number | null;
      balanced: boolean | null;
      subset_seed: number | null;
      synth_dim: number | null;
    }

    /** Description of the network topology (FlatMLP). */
    export interface TopologyDescriptor {
      hidden_sizes: number[];
      n_hidden_layers: number;
      n_classes: number | null;
      /** Single activation name or a per-layer list. */
      activation: string | string[] | null;
      /** Human-readable arch string, e.g. "x->128->64->10". */
      arch: string;
      l2: number | null;
    }

    /** Self-describing optimizer descriptor. */
    export interface OptimizerDescriptor {
      name: string;
      type: "sgd" | "adam" | "lbfgs" | "qqn" | string;
      learning_rate?: number | null;
      memory_size?: number | null;
      /** Optimizer-specific extra hyperparameters (e.g. QQN kwargs). */
      [key: string]: unknown;
    }

    /** Shared termination criteria (identical across optimizers). */
    export interface StopCriteria {
      f_target?: number | null;
      gtol?: number | null;
      time_budget?: number | null;
      milestones?: number[];
      [key: string]: unknown;
    }

    /**
     * A milestone hit tuple: [iteration, wall_time, evals, fwd, bwd].
     * `null` when the milestone was never crossed.
     */
    export type MilestoneHit =
      | [number, number, number | null, number | null, number | null]
      | null;

    /** The measured + derived output of a single optimizer run. */
    export interface RunResult {
      name: string;

      // Derived scalars.
      final_loss: number | null;
      best_loss: number | null;
      iters: number | null;
      train_acc: number | null;
      test_acc: number | null;
      wall: number;
      ms_per_iter: number | null;
      traj_auc: number | null;

      // Target / convergence accounting.
      iters_to_target: number | null;
      time_to_target: number | null;
      evals_to_target: number | null;
      evals_per_iter: number | null;
      reached: boolean;

      // Full trajectories (index-aligned by iteration).
      history: number[];
      times: number[];
      /** Cumulative combined value+grad evaluation counts per iteration. */
      eval_counts: number[] | null;
      /** Cumulative forward (value) evaluation counts per iteration. */
      fwd_counts: number[] | null;
      /** Cumulative backward (gradient) evaluation counts per iteration. */
      bwd_counts: number[] | null;

      /** Keyed by "%.6e"-formatted milestone loss value. */
      milestone_hits: Record<string, MilestoneHit>;
      /** Keyed by "%.6e"-formatted target loss value -> iteration hit. */
      target_iters: Record<string, number | null>;
    }

    /** A single-optimizer artifact (write_run_json). */
    export interface OptimizerRunArtifact {
      schema_version: SchemaVersion;
      kind: "optimizer_run";
      timestamp: string;
      dataset: DatasetDescriptor;
      topology: TopologyDescriptor;
      optimizer: OptimizerDescriptor;
      stop: StopCriteria | null;
      maxiter: number | null;
      seed: number | null;
      result: RunResult;
    }

    /** The aggregate experiment artifact (write_results_json). */
    export interface ExperimentArtifact {
      schema_version: SchemaVersion;
      kind: "experiment";
      timestamp: string;
      dataset: DatasetDescriptor;
      topology: TopologyDescriptor;
      /** Full resolved config (superset of dataset/topology fields). */
      config: Record<string, unknown>;
      results: Record<string, RunResult>;
      /** Optional merged extra metadata, e.g. axis analysis. */
      axis_analysis?: unknown;
      [key: string]: unknown;
    }

    export type ExperimentJson = OptimizerRunArtifact | ExperimentArtifact;