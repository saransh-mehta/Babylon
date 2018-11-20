'''
code to perform split on the giant trianing dataset
'''
import argparse
import numpy as np

# function to return number of lines in a  giant text file
def rawcount(filename):
    f = open(filename, 'rb')
    num_lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        num_lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return num_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type = str, help = 'path to training file')
    parser.add_argument('--output-name', type = str, help = 'prefix for output file name, training file is output_name_train.txt')
    parser.add_argument('--ratio', type = float, default = 0.1, help = 'split ratio (val/total)')
    parser.add_argument('--randomize', type = bool, default = True, help = 'to randomise data. if selected random points ')
    args = parser.parse_args()

    # get number of lines
    num_lines = rawcount(args.file_path)

    # perform splitting
    num_val = int(args.ratio * num_lines)

    print('[*] Total number of lines:', num_lines)
    print('[*] Total number of validation points:', num_val)

    # get validation indices
    all_indices = np.arange(num_lines)
    if args.randomize:
        np.random.shuffle(all_indices)

    # get training and validaton indices
    val_indices = all_indices[:num_val]
    train_indices = np.array([i for i in range(num_lines) if not i in val_indices])

    # open file and iterate
    f = open(args.file_path, 'r')
    ft = open(args.output_name + '_train.txt', 'w')
    fv = open(args.output_name + '_validation.txt', 'w')
    line_num = 0
    while line_num < num_lines:
        line = f.readline()
        if line_num in val_indices:
            fv.write(line)

        else:
            ft.write(line)

        line_num += 1

    # close the files
    f.close()
    ft.close()
    fv.close()

    print('[!] Training Data file saved at:', args.output_name + '_train.txt')
    print('[!] Validation Data file saved at:', args.output_name + '_validation.txt')

