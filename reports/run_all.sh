#!/bin/bash

# Clean
find results -name '*.json' -exec rm {} \;
find results -name '*.log' -exec rm {} \;
find results -name '*.png' -exec rm {} \;

# For each found run_reports.js, do to that directory and run it
#find . -name 'run_reports.js' -exec node {} \;
for run_report in $(find . -name 'run_reports.js'); do
    dir=$(dirname "$run_report")
    echo "Running report in $dir"
    (cd "$dir" && node run_reports.js)
done

# Clean experiment summaries (for clean manifest)
find . -name 'fashion_mnist_experiment_*.json' -exec rm {} \;

for result_dir in $(find . -name results); do
    dir="$result_dir"
    echo "Running report in $result_dir"
     (pushd "$dir" && \cp -f * ../../results/ && popd)
done

# Finish manifest
node generate-results-manifest.js