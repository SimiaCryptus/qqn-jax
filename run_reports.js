#!/usr/bin/env node
'use strict';

/*
 * run_reports.js
 *
 * Node.js replacement for run_reports.sh.
 *
 * Runs example benchmarks with timestamped log files and supports
 * named "variants" — predefined parameter sets (environment variables
 * and/or CLI args) for running the examples in standard configurations.
 *
 * Usage:
 *   node run_reports.js                      # run the default variant set
 *   node run_reports.js --list              # list all available variants
 *   node run_reports.js fashion_default     # run one or more named variants
 *   node run_reports.js fashion_mnist fashion_relu_deep
 *   node run_reports.js --all               # run every defined variant
 *   node run_reports.js --report fashion_mnist_mlp_comparison
 *                                           # run all variants of one report
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
//   report : the example module name under ./examples/<report>.py
//   env    : extra environment variables to set for the run
//   args   : extra CLI args to pass to the script
//   desc   : human-readable description
// ---------------------------------------------------------------------------

const REPORTS_DIR = './examples';
const RESULTS_DIR = 'results';
// ---------------------------------------------------------------------------
// Activation sweep configuration
//
// The canonical list of activation names mirrors the registry in
// experiments/models/activations.py. Each activation is exercised exactly
// once per report (sparse + comparison) with uniform parameters controlled
// by the global knobs below.
// ---------------------------------------------------------------------------
const ACTIVATION_TYPES = [
     'relu',
     'sigmoid',
     'sine',
     'gaussian',
     'triangle',
     'logabs',
     'tanh',
     'gelu',
     'swish',
     'softplus',
     'sawtooth',
     'abs',
     'identity',
     'rolling_sin',
     'rolling_atan2',
];
// Uniform parameters applied to every generated activation-sweep variant.
// Tune these once to change the whole sweep consistently.
const SWEEP_PARAMS = {
     N_TRAIN: '8000',
     N_TEST: '2000',
     HIDDEN: '128',
     DEPTH: '1',
     TIME_BUDGET: '30',   // seconds
     F_TARGET: '0.01',
};
// The reports to sweep and their default dataset.
const SWEEP_REPORTS = {
     comparison: {
         report: 'fashion_mnist_mlp_comparison',
         env: {DATASET: 'fashion_mnist'},
     },
     sparse: {
         report: 'mnist_sparse_benchmark',
         env: {DATASET: 'mnist'},
     },
};
// Build one variant per (report, activation) pair.
function buildActivationVariants() {
     const variants = {};
     for (const [tag, cfg] of Object.entries(SWEEP_REPORTS)) {
         for (const act of ACTIVATION_TYPES) {
             const name = `${tag}_act_${act}`;
             variants[name] = {
                 report: cfg.report,
                 env: {
                     ...SWEEP_PARAMS,
                     ...cfg.env,
                     ACTIVATION: act,
                 },
                 args: [],
                 desc:
                     `Activation sweep (${tag}): ${act} on ` +
                     `${cfg.report} with uniform sweep params.`,
             };
         }
     }
     return variants;
}


const VARIANTS = {};

Object.assign(VARIANTS, buildActivationVariants());
const DEFAULT_VARIANTS = //Object.keys(buildActivationVariants());
    [
        "comparison_act_relu",
        "comparison_act_sigmoid",
        "comparison_act_tanh",
        "comparison_act_sine",
        // "comparison_act_gaussian",
        // "comparison_act_triangle",
        // "comparison_act_logabs",
        // "comparison_act_gelu",
        // "comparison_act_swish",
        // "comparison_act_softplus",
        // "comparison_act_sawtooth",
        // "comparison_act_abs",
        // "comparison_act_identity",
        // "comparison_act_rolling_sin",
        // "comparison_act_rolling_atan2",
        // "sparse_act_relu",
        // "sparse_act_sigmoid",
        // "sparse_act_sine",
        // "sparse_act_gaussian",
        // "sparse_act_triangle",
        // "sparse_act_logabs",
        // "sparse_act_tanh",
        // "sparse_act_gelu",
        // "sparse_act_swish",
        // "sparse_act_softplus",
        // "sparse_act_sawtooth",
        // "sparse_act_abs",
        // "sparse_act_identity",
        // "sparse_act_rolling_sin",
        // "sparse_act_rolling_atan2"
    ];

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

function runVariant(name, variant) {
    return new Promise((resolve) => {
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
        if (Object.keys(variant.env).length) {
            console.log(`    env: ${JSON.stringify(variant.env)}`);
        }
        if (variant.args.length) {
            console.log(`    args: ${variant.args.join(' ')}`);
        }
        console.log(`    log: ${logfile}`);

        // Truncate ('w') rather than append: a fresh timestamped file per run
        // should never accumulate stale content.
        const logStream = fs.createWriteStream(logfile, {flags: 'w'});
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
        const child = spawn(executable, spawnArgs, {
            env: {...process.env, ...variant.env},
        });
       // Forward Ctrl-C to the child so a cancelled run tears down cleanly
       // instead of orphaning the Python/Scalene process.
       const onSigint = () => {
           console.error(`\n!!! Interrupted — terminating variant "${name}"`);
           child.kill('SIGTERM');
       };
       process.on('SIGINT', onSigint);
        // Write a reproducible header so each log is self-describing.
        const startedAt = Date.now();
        const envPairs = Object.entries(variant.env)
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
            .join(' ');
        logStream.write(
            `# variant: ${name}\n` +
            `# report:  ${variant.report}\n` +
            `# desc:    ${variant.desc}\n` +
            `# started: ${new Date(startedAt).toISOString()}\n` +
            `# command: ${envPairs ? envPairs + ' ' : ''}` +
            `${executable} ${spawnArgs.join(' ')}\n` +
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
        console.log(`  ${name.padEnd(width)}  [${v.report}]  ${v.desc}`);
    }
    console.log('\nDefault set:', DEFAULT_VARIANTS.join(', '));
   const reports = [
       ...new Set(Object.values(VARIANTS).map((v) => v.report)),
   ].sort();
   console.log('Reports     :', reports.join(', '));
}

function parseArgs(argv) {
    const opts = {list: false, all: false, report: null, variants: []};
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--list' || a === '-l') {
            opts.list = true;
        } else if (a === '--all' || a === '-a') {
            opts.all = true;
        } else if (a === '--report' || a === '-r') {
           opts.report = argv[++i];
           if (opts.report === undefined) {
               console.error('Error: --report requires a value.');
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
  node run_reports.js [variant ...]      Run named variant(s).
  node run_reports.js --all              Run every defined variant.
  node run_reports.js --report <name>    Run all variants for one report.
  node run_reports.js --list             List available variants.
  node run_reports.js --help             Show this help.

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

    ensureDir(RESULTS_DIR);
    ensureDir('profiles');

    let selected;
    if (opts.all) {
        selected = Object.keys(VARIANTS);
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

    let failures = 0;
    for (const name of selected) {
       const code = await runVariant(name, VARIANTS[name]);
        if (code !== 0) failures++;
    }

    console.log(
        `\nAll done. ${selected.length} variant(s) run, ${failures} failure(s).`
    );
    if (failures > 0) process.exitCode = 1;
}

main();