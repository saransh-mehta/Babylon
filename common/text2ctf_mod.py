'''
Heavily modified baseline text2ctf
'''

# importing the dependencies
import argparse
import re

# Initialize Global variables
glove_embeddings = {}

# The following method takes Glove Embedding file and stores all words and their embeddings in a dictionary
def load_embeddings(embedding_file):
    global glove_embeddings, emb_dim

    fe = open(embedding_file, 'r', encoding='utf-8', errors='ignore')
    for line in fe:
        tokens = line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = ' '.join(vec)
        glove_embeddings[word] = vec
    
    # Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    glove_embeddings['zerovec'] = '0.0 ' * emb_dim
    
    '''
    TODO: we can add embeddings for multiple 'unk' tokens to improve results
    '''

    fe.close()


def text_data_to_ctf_format(input_file, output_file, is_evaluation, op_mode):
    # get global values
    global glove_embeddings, emb_dim, max_query_words, max_passage_words

    lines_processed = 0

    f = open(input_file, 'r', encoding='utf-8', errors='ignore')
    fw = open(output_file, 'w', encoding='utf-8')
    for line in f:
        # break if 'SAMPLE'
        if op_mode == 'SAMPLE' and lines_processed == 3000:
            break

        # Format of the file : query_id \t query \t passage \t label \t passage_id
        tokens = line.strip().lower().split('\t')
        query_id, query, passage, label = tokens[0], tokens[1], tokens[2], tokens[3]

        # ****Query Processing****
        words = re.split('\W+', query)
        words = [x for x in words if x]  # to remove empty words
        word_count = len(words)
        remaining = max_query_words - word_count
        if (remaining > 0):
            words += ['zerovec'] * remaining  # Pad zero vecs if the word count is less than max_query_words
        words = words[:max_query_words]  # trim extra words
        # create Query Feature vector
        query_feature_vector = ''
        for word in words:
            if (word in glove_embeddings):
                query_feature_vector += glove_embeddings[word] + ' '
            else:
                query_feature_vector += glove_embeddings['zerovec'] + ' '  # Add zerovec for OOV terms
        query_feature_vector = query_feature_vector.strip()

        # ***** Passage Processing **********
        words = re.split('\W+', passage)
        words = [x for x in words if x]  # to remove empty words
        word_count = len(words)
        remaining = max_passage_words - word_count
        if (remaining > 0):
            words += ['zerovec'] * remaining  # Pad zero vecs if the word count is less than max_passage_words
        words = words[:max_passage_words]  # trim extra words
        # create Passage Feature vector
        passage_feature_vector = ''
        for word in words:
            if (word in glove_embeddings):
                passage_feature_vector += glove_embeddings[word] + ' '
            else:
                passage_feature_vector += glove_embeddings['zerovec'] + ' '  # Add zerovec for OOV terms
        passage_feature_vector = passage_feature_vector.strip()

        # convert label
        label_str = ' 1 0 ' if label == '0' else ' 0 1 '

        if (not is_evaluation):
            fw.write('|qfeatures ' + query_feature_vector + ' |pfeatures ' + passage_feature_vector + ' |labels ' + label_str + '\n')
        else:
            fw.write('|qfeatures ' + query_feature_vector + ' |pfeatures ' + passage_feature_vector + '|qid ' + str(query_id) + '\n')

        # increment proccessed lines
        lines_processed += 1

    # close the opened files
    f.close()
    fw.close()


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'operation mode, FULL for complete dump, SAMPLE for first 3000 lines')

    parser.add_argument('--train-file', type = str, help = 'path to training file')
    parser.add_argument('--valid-file', type = str, help = 'path to validation file')
    parser.add_argument('--eval-file', type = str, help = 'path to evaluation file')
    parser.add_argument('--glove-file', type = str, help = 'path to glove emdedding file')

    parser.add_argument('--max-query-len', type = int, default = 12, help = 'maximum length of query to be processed')
    parser.add_argument('--max-pass-len', type = int, default = 50, help = 'maximum length of passage to be processed')

    parser.add_argument('--prefix', type = str, help = 'prefix for this dump iteration')
    parser.add_argument('--verbose', type = bool, default = False, help = 'verbosity, (True for yes)')
    args = parser.parse_args()

    # set args
    max_query_words = args.max_query_len
    max_passage_words = args.max_pass_len
    emb_dim = int(args.glove_file.split('.')[-2].split('d')[0]) # auto detect

    # load the embeddings
    print('[*] loading the embeddings...')
    load_embeddings(args.glove_file)
    print('[*] ... loading complete')
    print('[!] number of words in embedding:', len(glove_embeddings))

    # Convert Query,Passage Text Data to CNTK Text Format(CTF)
    print('[*] training data conversion...')
    text_data_to_ctf_format(args.train_file, args.prefix + 'TrainData.ctf', False, args.mode)
    print('[*] ...train data conversion is done')

    print('[*] validation data conversion...')
    text_data_to_ctf_format(args.valid_file, args.prefix + 'ValidationData.ctf', False, args.mode)
    print('[*] ...validation data conversion is done')
    
    print('[*] evaluation data conversion ...')
    text_data_to_ctf_format(args.eval_file, args.prefix + 'EvaluationData.ctf', True, args.mode)
    print('[*] evaluation data conversion is done')
