#!/bin/bash

for i in {100..1600..50}
do
    python3 test_skin.py "$i" "$i" 100 100
done
