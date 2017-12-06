#!/bin/bash
for folder in subset2 subset3 subset4 subset5 subset6 subset7 subset8 subset9
        do
                python3 mask_extraction.py $folder
                python3 segment_lung.py $folder
        done

