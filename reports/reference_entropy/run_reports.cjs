#!/usr/bin/env node
'use strict';

/*
* run_reports.cjs
 *
 * Node.js replacement for run_reports.sh.
 *
 * Runs example benchmarks with timestamped log files and supports
 * named "variants" — predefined parameter sets (environment variables
 * and/or CLI args) for running the examples in standard configurations.
 *
 * Usage:
*   node run_reports.cjs                     # run the default variant set
*   node run_reports.cjs --list             # list all available variants
*   node run_reports.cjs eg_baseline        # run one or more named variants
*   node run_reports.cjs eg_h_mem_0p2 eg_tau_0p05
*   node run_reports.cjs --axis h_mem       # run every variant of one axis
*   node run_reports.cjs --all              # run every defined variant
*   node run_reports.cjs --report entropy_gated_benchmark
*                                           # run all variants of one report
*   node run_reports.cjs --dry-run ...      # print the commands only
*   node run_reports.cjs --env SEEDS=0,1 .. # override an env var for all runs
*   node run_reports.cjs --summarize        # results/*.log -> results/summary.csv
 */

const {spawn} = require('child_process');
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function timestamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return (
        `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
        `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
    );
}

function ensureDir(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, {recursive: true});
    }
}

// ---------------------------------------------------------------------------
// Variant definitions
//
// Each variant has:
//   report : the benchmark module name, ./<report>.py (REPORTS_DIR)
//   env    : extra environment variables to set for the run
//   args   : extra CLI args to pass to the script
//   desc   : human-readable description
//   scalene: (optional) true or an array of scalene args to profile the run
// ---------------------------------------------------------------------------

const REPORTS_DIR = './';
const RESULTS_DIR = 'results';
// Set by `--dry-run` / `--env K=V`; see parseArgs().
let DRY_RUN = false;
const ENV_OVERRIDES = {};
// Set by the SIGINT handler so the variant loop stops instead of launching
// the next cell after the current one has been killed.
let INTERRUPTED = false;










// ---------------------------------------------------------------------------
// EG-PTGP sweep configuration
//
// This report is about the *entropy gate*, not about activations or
// optimizer variants: every run uses the same model/activation and the
// standard QQN optimizer that entropy_gated_benchmark.py builds (L-BFGS
// oracle + Adam fallback, Armijo-Wolfe line search, no step-size memory).
// What we sweep are the gate / region hyper-parameters of
// docs/entropy_gated_regions.md §7.
// ---------------------------------------------------------------------------

// Shared data / model / optimizer knobs — identical across every variant so
// that only the gate parameters move.
const BASE_PARAMS = {
     DATASET: 'fashion_mnist',
     N_TRAIN: '5000',
     N_TEST: '2000',
     HIDDEN: '128',
     DEPTH: '2',
     ACTIVATION: 'tanh',
     MAXITER: '300',
     // Wall clock is a *safety net*, not the budget. A time budget truncates
     // cells at different iteration counts and every averaged metric then
     // measures "how many iterations did this cell get" (notes.md).
     // 0 disables it entirely; keep it generous otherwise.
     TIME_BUDGET: '0',
     LOG_EVERY: '20',
     LINE_SEARCH: 'armijo_wolfe',
     LBFGS_MEMORY: '50',
     ADAM_LR: '0.001',
     L2: '0.0001',
     SEED: '0',
     // Repeats: error bars, and the cross-seed churn H1 actually asks about.
     SEEDS: '0,1,2,3,4',
     DATA_SEED: '0',        // data split pinned so only init varies
     TAIL_ITERS: '100',     // window for tail churn / tail ECE
     FREEZE_PATIENCE: '25', // stop after N iters of t=0 and df~0 (H8 tell)
     FREEZE_TOL: '1e-9',
     REGION_DEBUG: '1',     // per-projection region instrumentation
};

// The operating point every one-factor-at-a-time sweep pivots around.
// policy=mean (k=1, dual exact in one sweep) is the pivot: it was the best
// cell on every axis in the first sweep, so the burden of proof is on k>1.
const EG_BASELINE = {
     H_MEM: '0.5',
     TAU: '0.1',
     KAPPA: '0.1',
     ZETA: '0.0',
     POLICY: 'mean',
     NUM_REGIONS: '8',
     MAX_CONSTRAINTS: '256',
     GRAD_CHUNK: '64',
     WARMUP: '5',
};

// One-factor-at-a-time axes. `extraEnv` pins any companion knob that only
// makes sense for that axis (e.g. NUM_REGIONS only bites for clustered or
// random partitions).
const SWEEP_AXES = {
     h_mem: {
         env: 'H_MEM',
         // 0.3–0.6 is the band where the effect exists and the freeze has not
         // set in; the extremes are kept only to re-confirm the two dead-zone
         // regimes (inert gate below ~0.2, hard freeze above ~0.7).
         values: ['0.1', '0.2', '0.3', '0.35', '0.4', '0.5', '0.6', '0.7', '1.0'],
         desc: 'memorization entropy threshold H_mem (nats)',
     },
     tau: {
         env: 'TAU',
         values: ['0.02', '0.05', '0.1', '0.25', '0.5'],
         desc: 'gate band width tau (beta = 1/tau)',
     },
     kappa: {
         env: 'KAPPA',
         values: ['0.0', '0.05', '0.1', '0.25', '0.5'],
         desc: 'slack coefficient kappa',
     },
     zeta: {
         env: 'ZETA',
         values: ['0.0', '0.25', '0.5', '1.0'],
         desc: 'dispersion tightening zeta',
     },
     policy: {
         env: 'POLICY',
         values: ['mean', 'class', 'pair', 'gradient', 'sample', 'random'],
         desc: 'region policy',
     },
     num_regions: {
         env: 'NUM_REGIONS',
         values: ['2', '4', '8', '16', '32'],
         extraEnv: {POLICY: 'gradient'},
         desc: 'number of gradient-space regions k (POLICY=gradient)',
     },
     max_constraints: {
         env: 'MAX_CONSTRAINTS',
         values: ['64', '128', '256', '512', '1024'],
         desc: 'cap on |M| per step',
     },
     warmup: {
         env: 'WARMUP',
         values: ['0', '5', '20', '50'],
         desc: 'unprojected warmup steps',
     },
};

// Filenames must stay shell/regex friendly: 0.35 -> 0p35, -1 -> m1.
function sanitizeValue(v) {
     return String(v).replace(/\./g, 'p').replace(/^-/, 'm');
}

function buildSweepVariants() {
     const variants = {};
     variants['eg_baseline'] = {
         report: 'entropy_gated_benchmark',
         axis: 'baseline',
         env: {...BASE_PARAMS, ...EG_BASELINE},
         args: [],
         desc: 'EG-PTGP ablation (4 cells) at the baseline operating point.',
     };
     for (const [axis, cfg] of Object.entries(SWEEP_AXES)) {
         for (const value of cfg.values) {
             const name = `eg_${axis}_${sanitizeValue(value)}`;
             variants[name] = {
                 report: 'entropy_gated_benchmark',
                 axis,
                 env: {
                     ...BASE_PARAMS,
                     ...EG_BASELINE,
                     ...(cfg.extraEnv || {}),
                     [cfg.env]: String(value),
                 },
                 args: [],
                 desc: `Sweep ${cfg.desc}: ${cfg.env}=${value}.`,
             };
         }
     }
     return variants;
}

const VARIANTS = {};

Object.assign(VARIANTS, buildSweepVariants());

function variantsForAxis(axis) {
     return Object.keys(VARIANTS).filter((n) => VARIANTS[n].axis === axis);
}

// Default set: the baseline cell plus the axes that decide whether the gate
// has any content at all (where the gate sits, how sharp it is, and how the
// constraints are grouped). Run `--axis kappa` etc. for the rest.
// NOTE: every cell now runs SEEDS in-process, so a "variant" is already
// 4 cells x |SEEDS| runs — budget accordingly.
const DEFAULT_VARIANTS = [
     'eg_baseline',
     'eg_h_mem_0p3',
     'eg_h_mem_0p4',
     'eg_h_mem_0p6',
     ...variantsForAxis('tau'),
     ...variantsForAxis('policy'),
];

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

function runVariant(name, variant) {
    return new Promise((resolve) => {
        // CLI `--env K=V` wins over the variant's own settings so a one-off
        // sweep never requires editing this file.
        const runEnv = {...variant.env, ...ENV_OVERRIDES};
        // Use a per-variant timestamp so sequential runs are distinguishable
        // and never silently collide.
        const variantTs = timestamp();
        const logfile = path.join(
            RESULTS_DIR,
            `${variant.report}_${name}_${variantTs}.log`
        );
        const scriptPath = path.join(REPORTS_DIR, `${variant.report}.py`);
        if (!fs.existsSync(scriptPath)) {
            console.error(
                `!!! Script not found for variant "${name}": ${scriptPath}`
            );
            resolve(1);
            return;
        }


        console.log(`\n=== Running variant "${name}" (${variant.report}) ===`);
        console.log(`    ${variant.desc}`);
        if (Object.keys(runEnv).length) {
            console.log(`    env: ${JSON.stringify(runEnv)}`);
        }
        if (variant.args.length) {
            console.log(`    args: ${variant.args.join(' ')}`);
        }
        console.log(`    log: ${logfile}`);

        // If the variant requests scalene execution, use scalene as the
        // launcher so it actually captures a profile rather than just
        // printing a hint.
        let executable, spawnArgs;
        if (variant.scalene) {
            const profileArgs = variant.scalene === true ? [] : variant.scalene;
            executable = 'scalene';
            // Scalene >= 2.3.0 uses a subcommand-based CLI: `scalene run <script>`.
            // Use `--` to clearly separate Scalene options from the script's own args.
            spawnArgs = ['run', ...profileArgs, '--', scriptPath, ...variant.args];
        } else {
            executable = 'python3';
            spawnArgs = [scriptPath, ...variant.args];
        }
        // Surface the exact command line (including any env overrides) on the
        // console so runs are reproducible without needing to open the log.
        const envPairs = Object.entries(runEnv)
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
            .join(' ');
        const cmdLine =
            `${envPairs ? envPairs + ' ' : ''}` +
            `${executable} ${spawnArgs.join(' ')}`;
        console.log(`    cmd: ${cmdLine}`);
        if (DRY_RUN) {
            console.log('    (dry run — nothing executed, no log written)');
            resolve(0);
            return;
        }
        // Truncate ('w') rather than append: a fresh timestamped file per run
        // should never accumulate stale content.
        const logStream = fs.createWriteStream(logfile, {flags: 'w'});
        const child = spawn(executable, spawnArgs, {
            env: {...process.env, ...runEnv},
        });
       // Forward Ctrl-C to the child so a cancelled run tears down cleanly
       // instead of orphaning the Python/Scalene process.
       const onSigint = () => {
           console.error(`\n!!! Interrupted — terminating variant "${name}"`);
           INTERRUPTED = true;
           child.kill('SIGTERM');
       };
       process.on('SIGINT', onSigint);
        // Write a reproducible header so each log is self-describing.
        const startedAt = Date.now();
        logStream.write(
            `# variant: ${name}\n` +
            `# report:  ${variant.report}\n` +
            `# desc:    ${variant.desc}\n` +
            `# started: ${new Date(startedAt).toISOString()}\n` +
            `# command: ${cmdLine}\n` +
            `${'-'.repeat(72)}\n`
        );


        // Tee stdout/stderr to both the console and the log file.
        child.stdout.on('data', (data) => {
            process.stdout.write(data);
            logStream.write(data);
        });
        child.stderr.on('data', (data) => {
            process.stderr.write(data);
            logStream.write(data);
        });

        child.on('close', (code) => {
           process.removeListener('SIGINT', onSigint);
            const elapsedS = ((Date.now() - startedAt) / 1000).toFixed(1);
            logStream.write(
                `\n${'-'.repeat(72)}\n` +
                `# exit code: ${code}  elapsed: ${elapsedS}s\n`
            );
            logStream.end();
            if (code !== 0) {
                console.error(
                    `!!! variant "${name}" exited with code ${code} (${elapsedS}s)`
                );
            } else {
                console.log(`=== Finished variant "${name}" (${elapsedS}s) ===`);
            }
            resolve(code);
        });

        child.on('error', (err) => {
           process.removeListener('SIGINT', onSigint);
            console.error(`!!! Failed to start variant "${name}": ${err.message}`);
            logStream.end();
            resolve(1);
        });
    });
}

function listVariants() {
    console.log('Available variants:\n');
    const names = Object.keys(VARIANTS);
    const width = Math.max(...names.map((n) => n.length));
    for (const name of names) {
        const v = VARIANTS[name];
         console.log(
             `  ${name.padEnd(width)}  [${v.axis}]  ${v.desc}`
         );
    }
    console.log('\nDefault set:', DEFAULT_VARIANTS.join(', '));
   const reports = [
       ...new Set(Object.values(VARIANTS).map((v) => v.report)),
   ].sort();
   console.log('Reports     :', reports.join(', '));
    const axes = [
        ...new Set(Object.values(VARIANTS).map((v) => v.axis)),
    ].sort();
    console.log('Axes        :', axes.join(', '));
}
// ---------------------------------------------------------------------------
// Aggregation: turn the `[row]` footers of every log into one CSV.
// ---------------------------------------------------------------------------
function summarize(outPath) {
    if (!fs.existsSync(RESULTS_DIR)) {
        console.error(`No results directory at ${RESULTS_DIR}.`);
        process.exitCode = 1;
        return;
    }
    const files = fs
        .readdirSync(RESULTS_DIR)
        .filter((f) => f.endsWith('.log'))
        .sort();
    const rows = [];
    const keys = [];
    const seenKeys = new Set();
    for (const file of files) {
        const text = fs.readFileSync(path.join(RESULTS_DIR, file), 'utf8');
        for (const line of text.split('\n')) {
            const i = line.indexOf('[row]');
            if (i < 0) continue;
            const row = {log: file};
            for (const tok of line.slice(i + 5).trim().split(/\s+/)) {
                const eq = tok.indexOf('=');
                if (eq <= 0) continue;
                row[tok.slice(0, eq)] = tok.slice(eq + 1);
            }
            for (const k of Object.keys(row)) {
                if (!seenKeys.has(k)) {
                    seenKeys.add(k);
                    keys.push(k);
                }
            }
            rows.push(row);
        }
    }
    if (!rows.length) {
        console.error('No "[row]" lines found in results/*.log.');
        process.exitCode = 1;
        return;
    }
    const esc = (v) =>
        /[",\n]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
    const csv = [keys.join(',')]
        .concat(rows.map((r) => keys.map((k) => esc(r[k] ?? '')).join(',')))
        .join('\n');
    fs.writeFileSync(outPath, csv + '\n');
    console.log(
        `Wrote ${rows.length} row(s) from ${files.length} log(s) to ${outPath}.`
    );
}


function parseArgs(argv) {
     const opts = {
         list: false, all: false, help: false, report: null, axis: null,
         dryRun: false, summarize: null, variants: [],
     };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--list' || a === '-l') {
            opts.list = true;
        } else if (a === '--all' || a === '-a') {
            opts.all = true;
        } else if (a === '--dry-run' || a === '-n') {
            opts.dryRun = true;
        } else if (a === '--summarize' || a === '-s') {
            opts.summarize = path.join(RESULTS_DIR, 'summary.csv');
        } else if (a === '--env' || a === '-e') {
            const kv = argv[++i];
            const eq = kv === undefined ? -1 : kv.indexOf('=');
            if (eq <= 0) {
                console.error('Error: --env requires KEY=VALUE.');
                process.exit(1);
            }
            ENV_OVERRIDES[kv.slice(0, eq)] = kv.slice(eq + 1);
        } else if (a === '--report' || a === '-r') {
           opts.report = argv[++i];
           if (opts.report === undefined) {
               console.error('Error: --report requires a value.');
               process.exit(1);
           }
         } else if (a === '--axis' || a === '-x') {
            opts.axis = argv[++i];
            if (opts.axis === undefined) {
                console.error('Error: --axis requires a value.');
                process.exit(1);
            }
        } else if (a === '--help' || a === '-h') {
            opts.help = true;
        } else {
            opts.variants.push(a);
        }
    }
    return opts;
}

function printHelp() {
    console.log(`run_reports.js — run example benchmarks with named variants.

Usage:
  node run_reports.cjs [variant ...]     Run named variant(s).
  node run_reports.cjs --all             Run every defined variant.
  node run_reports.cjs --axis <name>     Run all variants of one sweep axis.
  node run_reports.cjs --report <name>   Run all variants for one report.
  node run_reports.cjs --list            List available variants.
  node run_reports.cjs --dry-run ...     Print commands without running them.
  node run_reports.cjs --env K=V ...     Override an env var for every run
                                         (repeatable; e.g. --env SEEDS=0,1).
  node run_reports.cjs --summarize       Aggregate every "[row]" line in
                                         results/*.log into results/summary.csv.
  node run_reports.cjs --help            Show this help.

With no arguments, runs the default set: ${DEFAULT_VARIANTS.join(', ')}.
`);
}

async function main() {
    const opts = parseArgs(process.argv.slice(2));

    if (opts.help) {
        printHelp();
        return;
    }
    if (opts.list) {
        listVariants();
        return;
    }
    if (opts.summarize) {
        summarize(opts.summarize);
        return;
    }
    DRY_RUN = opts.dryRun;

    ensureDir(RESULTS_DIR);

    let selected;
    if (opts.all) {
        selected = Object.keys(VARIANTS);
     } else if (opts.axis) {
         selected = variantsForAxis(opts.axis);
         if (selected.length === 0) {
             const axes = [
                 ...new Set(Object.values(VARIANTS).map((v) => v.axis)),
             ].sort();
             console.error(`No variants found for axis "${opts.axis}".`);
             console.error(`Known axes: ${axes.join(', ')}`);
             process.exitCode = 1;
             return;
         }
    } else if (opts.report) {
        selected = Object.keys(VARIANTS).filter(
            (n) => VARIANTS[n].report === opts.report
        );
        if (selected.length === 0) {
            const reports = [
                ...new Set(Object.values(VARIANTS).map((v) => v.report)),
            ].sort();
            console.error(`No variants found for report "${opts.report}".`);
            console.error(`Known reports: ${reports.join(', ')}`);
            process.exitCode = 1;
            return;
        }
    } else if (opts.variants.length) {
        selected = opts.variants;
    } else {
        selected = DEFAULT_VARIANTS;
    }

    // Validate selection.
    const unknown = selected.filter((n) => !VARIANTS[n]);
    if (unknown.length) {
        console.error(`Unknown variant(s): ${unknown.join(', ')}`);
        console.error('Use --list to see available variants.');
        process.exitCode = 1;
        return;
    }
   // De-duplicate while preserving order so an accidental repeat does not
   // run (and overwrite logs for) the same variant twice.
   const seen = new Set();
   const deduped = selected.filter((n) => {
       if (seen.has(n)) return false;
       seen.add(n);
       return true;
   });
   if (deduped.length !== selected.length) {
       console.warn(
           `(note: ignored ${selected.length - deduped.length} duplicate ` +
           `variant selection(s))`
       );
   }
   selected = deduped;
    // Only create a profiles/ directory if something will actually profile.
    if (selected.some((n) => VARIANTS[n].scalene)) {
        ensureDir('profiles');
    }


    let failures = 0;
    for (const name of selected) {
       const code = await runVariant(name, VARIANTS[name]);
        if (code !== 0) failures++;
        if (INTERRUPTED) {
            console.error('!!! Interrupted — skipping remaining variants.');
            process.exitCode = 130;
            return;
        }
    }

    console.log(
        `\nAll done. ${selected.length} variant(s) run, ${failures} failure(s).`
    );
    if (failures > 0) process.exitCode = 1;
}

main();