#!/usr/bin/bash

for i in ../../DAT/*.dat; do
    bn=`basename $i .dat`
    ./draw_img.py 128 128 $i > ../../PPM/${bn}.ppm
done