#!/bin/bash

set -e
echo "Checking everything is ready"
python3 init.py --config ${1:-"Default"}
echo "Starting voice generation"
python3 voiceGenerator.py --config ${1:-"Default"} --step ${2:-"1"}
echo "Done"