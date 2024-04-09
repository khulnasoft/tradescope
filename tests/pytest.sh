#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=tradescope --cov-config=.coveragerc tests/
