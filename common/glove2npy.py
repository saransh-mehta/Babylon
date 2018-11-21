'''
Script to convert the input embeddings to numpy array dumps
'''

import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type = str, help = 'path to training file')
    parser.add_argument('--output-name', type = str,
        help = 'data file is output_name.npy, word file as output_name_words.txt')
    args = parser.parse_args()

    # open the file
    f = open(args.file_path, 'rb')

    # buffer matrices
    embeddings_ = []
    all_words = []

    num_lines_processed = 0

    while True:
        num_lines_processed += 1
        line = f.readline()
        if not line:
            break

        tokens = line.split(b'\n')[0].split(b' ')
        word = tokens[0]
        embd = np.array([np.float32(i) for i in tokens[1:]])
        all_words.append(str(word) + '|')
        embeddings_.append(embd)

        # print(line_WE)

        '''
        if len(embeddings_) == 0:
            embeddings_ = embd

        else:
            embeddings_ = np.append(embeddings_, embd)
        '''

    # save the data 
    print('Number of buffers handled:', num_lines_processed)
    np.save(args.output_name + '.npy', np.array(embeddings_))
    del embeddings_
    
    # save the words
    words = ' '.join(all_words)
    f = open(args.output_name + '_words.txt', 'w')
    f.write(words)
    f.close()
