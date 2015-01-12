# Vector Generator
Generates vectors and saves them to `.vec` files. 

## Synopsis
usage: `vecGen.py [-h] [-v N [N ...]] [-s] [-r R]`

Generate sorted and shuffled vectors

Arguments:
  `-h, --help`          show this help message and exit
  `-v N [N ...], --vector-lengths N [N ...]`
                        A set of vector lenghts.
  `-s, --shuffle`       Shuffle the vectors (`True` or `False`). Defaults to False.
  `-r R, --random-seed R`
                        The seed to use when shuffling the vectors. Defaults to the system time. 

