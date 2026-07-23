// Generate results/manifest.json to index all optimizer results in results/
//
// Usage:
//   node generate-results-manifest.js [resultsDir]
//
// Scans the results directory for *.json artifacts (as defined by
// experiments/reporting/schema.ts), extracts light-weight summary metadata
// for each, and writes results/manifest.json describing the index.

'use strict';

const fs = require('fs');
const path = require('path');

const SCHEMA_VERSION = '1.0.0';

/**
 * Recursively collect *.json files under a directory (excluding manifest.json).
 * @param {string} dir
 * @param {string} baseDir
 * @returns {string[]} relative paths (posix separators)
 */
function collectJsonFiles(dir, baseDir) {
    const out = [];
    let entries;
    try {
        entries = fs.readdirSync(dir, {withFileTypes: true});
    } catch (err) {
        console.error(`Cannot read directory ${dir}: ${err.message}`);
        return out;
    }

    for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            out.push(...collectJsonFiles(full, baseDir));
        } else if (entry.isFile() && entry.name.endsWith('.json')) {
            if (entry.name === 'manifest.json') continue;
            const rel = path.relative(baseDir, full).split(path.sep).join('/');
            out.push(rel);
        }
    }
    return out;
}

/**
 * Read + parse a JSON artifact, returning null on failure.
 * @param {string} file
 * @returns {object | null}
 */
function readJson(file) {
    try {
        const raw = fs.readFileSync(file, 'utf8');
        return JSON.parse(raw);
    } catch (err) {
        console.warn(`Skipping ${file}: ${err.message}`);
        return null;
    }
}

/**
 * Summarize a single RunResult into compact form (no full trajectories).
 * @param {object} r
 * @returns {object}
 */
function summarizeRun(r) {
    if (!r || typeof r !== 'object') return null;
    return {
        name: r.name ?? null,
        final_loss: r.final_loss ?? null,
        best_loss: r.best_loss ?? null,
        iters: r.iters ?? null,
        train_acc: r.train_acc ?? null,
        test_acc: r.test_acc ?? null,
        wall: r.wall ?? null,
        ms_per_iter: r.ms_per_iter ?? null,
        traj_auc: r.traj_auc ?? null,
        iters_to_target: r.iters_to_target ?? null,
        time_to_target: r.time_to_target ?? null,
        evals_to_target: r.evals_to_target ?? null,
        reached: r.reached ?? false,
        history_len: Array.isArray(r.history) ? r.history.length : 0,
    };
}

/**
 * Build a manifest entry for one artifact file.
 * @param {string} relPath
 * @param {object} json
 * @param {fs.Stats} stat
 * @returns {object}
 */
function buildEntry(relPath, json, stat) {
    const kind = json.kind;
    const base = {
        path: relPath,
        kind: kind ?? 'unknown',
        schema_version: json.schema_version ?? null,
        timestamp: json.timestamp ?? null,
        mtime: stat.mtimeMs,
        size: stat.size,
        dataset: json.dataset ?? null,
        topology: json.topology
            ? {
                  arch: json.topology.arch ?? null,
                  l2: json.topology.l2 ?? null,
                  activation: json.topology.activation ?? null,
              }
            : null,
    };

    if (kind === 'optimizer_run') {
        return {
            ...base,
            optimizer: json.optimizer
                ? {name: json.optimizer.name, type: json.optimizer.type}
                : null,
            seed: json.seed ?? null,
            maxiter: json.maxiter ?? null,
            run: summarizeRun(json.result),
        };
    }

    if (kind === 'experiment') {
        const results = json.results && typeof json.results === 'object' ? json.results : {};
        const runNames = Object.keys(results);
        return {
            ...base,
            n_runs: runNames.length,
            runs: runNames.map((name) => summarizeRun(results[name])),
        };
    }

    return base;
}

function main() {
    const resultsDir = process.argv[2]
        ? path.resolve(process.argv[2])
        : path.resolve(process.cwd(), 'results');

    if (!fs.existsSync(resultsDir)) {
        console.error(`Results directory not found: ${resultsDir}`);
        process.exit(1);
    }

    const files = collectJsonFiles(resultsDir, resultsDir);
    const entries = [];

    for (const rel of files) {
        const full = path.join(resultsDir, rel);
        const json = readJson(full);
        if (!json) continue;
        let stat;
        try {
            stat = fs.statSync(full);
        } catch (err) {
            console.warn(`Cannot stat ${full}: ${err.message}`);
            continue;
        }
        entries.push(buildEntry(rel, json, stat));
    }

    // Newest first.
    entries.sort((a, b) => (b.mtime ?? 0) - (a.mtime ?? 0));

    const manifest = {
        schema_version: SCHEMA_VERSION,
        generated_at: new Date().toISOString(),
        count: entries.length,
        entries,
    };

    const outPath = path.join(resultsDir, 'manifest.json');
    fs.writeFileSync(outPath, JSON.stringify(manifest, null, 2));
    console.log(`Wrote ${outPath} (${entries.length} entries).`);
}

main();