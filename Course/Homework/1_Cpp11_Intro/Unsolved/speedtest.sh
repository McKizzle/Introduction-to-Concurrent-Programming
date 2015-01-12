#!/bin/bash

for max_threads in {1..20}
do
    time ./syncron $max_threads 100000 0
done
