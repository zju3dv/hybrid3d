#!/usr/bin/env bash
# conda activate pytorch
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
