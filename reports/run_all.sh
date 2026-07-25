#!/bin/bash

# Clean
find results -name '*.json' -exec rm {} \;
find results -name '*.log' -exec rm {} \;
find results -name '*.png' -exec rm {} \;

# Run
find . -name 'run_reports.js' -exec node {} \;

# Clean experiment summaries (for clean manifest)
find . -name 'fashion_mnist_experiment_*.json' -exec rm {} \;
# Finish manifest
node generate-results-manifest.js

