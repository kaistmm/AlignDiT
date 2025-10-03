#!/bin/bash

FILE=$(realpath "$0" | sed 's|/run/|/script/|g' | sed 's/\.sh$/.py/')

OMP_NUM_THREADS=1 \
PYTHONPATH=src \
python $FILE \
