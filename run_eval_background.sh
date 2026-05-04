#!/bin/bash
conda run -n steer python src/steering.py > eval.log 2>&1
