'''
Stripped down transformer network inspired architecture for MSAIC 2018
Built by Yash Bonde for team msaic_yash_sara_kuma_6223

This file contains the class for the network architecture, with built in functions for training
and operations. Any crucial step will be commented there, for more information on the model
read blog at: 
and:

Cheers!
'''

# importng the dependencies
import numpy as np
import core_layers # major layer and building blocks for the model
import model_config as cfg

# class

class TransformerNetwork(object):
    """
    [GOTO: https://stackoverflow.com/a/35688187]

    The idea is that since we have an external embedding matrix, we can still use the
    functionalities available in TF to use those embedding. This will require us to store the
    embedding matrix in memory and then assign it at runtime. Function assign_embeddings() does it.

    Update: all the architecture related values have been pushed to the model_config file
    """
    def __init__(self, scope):
        self.scope = scope

        # declaring the placeholders
        self.q_plhdr = tf.placeholder(tf.float32, [None, cfg.DIM_MODEL], name = 'query_placeholder')
        self.p_plhdr = tf.placeholder(tf.float32, [None, cfg.DIM_MODEL], name = 'passage_placeholder')
        self.target_plhdr = tf.placeholder(tf.float32, [None, 1], name = 'target_placeholder')

        # operation values
        self.global_step = 0

        # build network
        self.build_model()

    def build_model(self):
        '''
        function to build the model end to end
        '''
        with tf.variable_scope(self.scope):
            q_out = self.q_plhdr
            p_out = self.p_plhdr
            for i in range(cfg.NUM_STACKS):
                q_out = core_layers.query_stack(q_in = q_out, mask = input_mask, scope = 'q_stk_{i}')
                p_out = core_layers.passage_stack(p_in = p_out, q_out = q_out,
                    input_mask = input_mask, target_mask = target_mask, scope = 'p_stk_{i}')

            # now the custom part
            ff_out = tf.layers.dense(p_out, cfg.FINAL_FF_MID1, activation = tf.nn.relu)
            ff_out = tf.layers.dense(ff_out, cfg.FINAL_FF_MID2)
            self.pred = tf.layers.dense(ff_out, 1, activation = tf.nn.tanh)

    def construct_padding_mask(self, inp):
        '''
        Args:
            inp: Original input of word ids, shape: [batch_size, seqlen]
        Returns:
            a mask of shape [batch_size, seqlen, seqlen] where <pad> is 0 and others are 1
        '''
        seqlen = inp.shape.as_list()[1]
        mask = tf.cast(tf.not_equal(inp, self.pad_id), tf.float32)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seqlen, 1])
        return mask

    def assign_embeddings(self, emb):
        '''
        assign_embeddings uses external embedding matrix and assigns it to the embedding matrix here
        Args:
            emb: a numpy array of shape [num_words, e_dim]
        '''
        self.embedding_matrix = tf.Variable(tf.constant(0.0, shape = emb.shape, trainable = False,
                                            name = 'embedding_matrix'))
        embedding_placeholder = tf.placeholder(tf.float32, shape = emb.shape)
        embedding_init = self.embedding_matrix.assign(embedding_placeholder)
        self.sess.run(embedding_init, feed_dict: {embedding_placeholder: emb})

























        
