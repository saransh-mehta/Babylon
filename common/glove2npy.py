'''
Script to convert the input embeddings to numpy array dumps
'''

import argparse
import numpy as np

def read_buffers(filename):
	f = open(filename)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type = str, help = 'path to training file')
    parser.add_argument('--output-name', type = str,
    	help = 'prefix for output file name, training file is output_name.npy')
    args = parser.parse_args()

    # open the file
    