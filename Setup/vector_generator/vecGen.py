#!/usr/bin/env python

import random as rndm
import argparse as agpe
import time as tm
import sys as sys
import os as os

def main(vector_lengths, shuffle, seed, delimiter):
    rndm.seed(seed)

    vectors = [ range(0, vlen) for vlen in vector_lengths ]

    golden_dir = "golden" 
    if not os.path.exists(golden_dir):
        os.makedirs(golden_dir)
    
    [ vector2file("%s/%i.vec" % (golden_dir, len(vector)), vector) for vector in vectors ]

    if(shuffle):
        shuffle_dir = "shuffled"
        if not os.path.exists(shuffle_dir):
            os.makedirs(shuffle_dir)

        [ rndm.shuffle(vector) for vector in vectors ] 
        [ vector2file("%s/%i.vec" % (shuffle_dir, len(vector)), vector)  for vector in vectors ] 

    return 0

def vector2file(file_name, vector, delimiter=','):
    with open(file_name, "w") as vector_file:
        for i in vector:
            vector_file.write("%i%s" % (i, delimiter))

if __name__=="__main__":
    parser = agpe.ArgumentParser(description="Generate sorted and shuffled vectors")
    parser.add_argument("-d", "--delimiter", type=str, metavar="D", default=',', help="Default delimiter to use");
    parser.add_argument("-v", "--vector-lengths", type=int, nargs='+', metavar='N', help="A set of vector lenghts. ") 
    parser.add_argument("-s", "--shuffle", action="store_true", help="Shuffle the vectors [True / False]. Defaults to false.")
    parser.add_argument("-r", "--random-seed", type=int, metavar="R", default=tm.time(), help="The seed to use when shuffling the vectors. Defaults to the system time")
    args = parser.parse_args()

    if(len(sys.argv) >= 2):
        main(args.vector_lengths, args.shuffle, args.random_seed, args.delimiter)
    else:
        parser.print_help()

