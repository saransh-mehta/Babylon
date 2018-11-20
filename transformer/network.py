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


cfg.DIM_MODEL = dim_out

class TransformerNetwork(object):
    """
    Update: all the architecture related values have been pushed to he model_config file
    """
    def __init__(self, scope):
        self.scope = scope

        # declaring the placeholders
        self.q_plhdr = tf.placeholder(tf.float32, [None, cfg.DIM_MODEL], name = 'query_placeholder')
        self.p_plhdr = tf.placeholder(tf.float32, [None, cfg.DIM_MODEL], name = 'passage_placeholder')
        self.target_plhdr = tf.placeholder(tf.float32, [None, 1], name = 'target_placeholder')

        # operation values
        self.global_step = 0

    def build_model(self):
        '''
        function to build the model end to end
        '''
        with tf.variable_scope(self.scope):
            q_out = self.q_plhdr
            p_out = self.p_plhdr
            for i in range(self.num_stacks):
                q_out = core_layers.query_stack(q_out, mask, scope)
                p_out = core_layers.passage_stack(p_out, q_out, input_mask, target_mask, scope)
        
