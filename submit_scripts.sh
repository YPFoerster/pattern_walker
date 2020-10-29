#!/bin/bash

for filename in redraw_patterns_combi_o_g/.job/*.job; do
    sbatch "$filename"
done
