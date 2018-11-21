'''
Stripped down transformer network inspired architecture for MSAIC 2018
Built by Yash Bonde for team msaic_yash_sara_kuma_6223

This file contains the util for the network architecture. Any crucial step will be commented
there, for more information on the model
read blog at: 
and:

Cheers!
'''

# importing the dependencies 
import model_config as cfg

# functions
def sdpa(Q, K, V, mask = None):
    '''
    Scaled Dot Product Attention
    q_size = k_size = v_size
    Args:
        Q:    (num_heads * batch_size, q_size, d_model)
        K:    (num_heads * batch_size, k_size, d_model)
        V:    (num_heads * batch_size, v_size, d_model)
        mask: (num_heads * batch_size, q_size, d_model)
    '''

    qkt = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    qkt /= tf.sqrt(np.float32(cfg.DIM_MODEL // cfg.NUM_HEADS))

    if mask:
        # perform masking
        qkt = tf.multiply(qkt, mask) + (1.0 - mask) * (-1e10)

    soft = tf.nn.softmax(qkt) # (num_heads * batch_size, q_size, k_size)
    soft = tf.layers.dropout(soft, training = cfg.OP_TRAINING)
    out = tf.matmul(soft, V) # (num_heads * batch_size, q_size, d_model)

    return out

def multihead_attention(query, key, value, mask = None, scope = 'attention'):
    '''
    Multihead attention with masking option
    q_size = k_size = v_size = d_model/num_heads
    Args:
        query: (batch_size, q_size, d_model)
        key:   (batch_size, k_size, d_model)
        value: (batch_size, v_size, d_model)
        mask:  (batch_size, q_size, d_model)
    '''
    with tf.variable_scope(scope):
        # linear projection blocks
        Q = tf.layers.dense(query, cfg.DIM_MODEL, activation = tf.nn.relu)
        K = tf.layers.dense(key, cfg.DIM_MODEL, activation = tf.nn.relu)
        V = tf.layers.dense(value, cfg.DIM_MODEL, activation = tf.nn.relu)

        # split the matrix into multiple heads and then concatenate them to get
        # a larger batch size: (num_heads, q_size, d_model/nume_heads)
        Q_reshaped = tf.concat(tf.split(Q, cfg.NUM_HEADS, axis = 2), axis = 0)
        K_reshaped = tf.concat(tf.split(K, cfg.NUM_HEADS, axis = 2), axis = 0)
        V_reshaped = tf.concat(tf.split(V, cfg.NUM_HEADS, axis = 2), axis = 0)
        if mask:
            mask = tf.tile(mask, [cfg.NUM_HEADS, 1, 1])

        # scaled dot product attention
        sdpa_out = sdpa(Q_reshaped, K_reshaped, V_reshaped, mask)
        out = tf.concat(tf.split(sdpa_out, cfg.NUM_HEADS, axis = 0), axis = 2)

        # final linear layer
        out_linear = tf.layers.dense(out, cfg.DIM_MODEL)
        out_linear = tf.layers.dropout(out_linear, training = cfg.OP_TRAINING)

    return out_linear

def feed_forward(x):
    '''
    Position-wise feed forward network, applied to each position seperately
    and identically. Can be implemented as follows
    '''
    with tf.variable_scope('ff'):
        out = tf.layers.conv1d(x, filters = cfg.FF_MID, kernel_size = 1,
            activation = tf.nn.relu)
        out = tf.layers.conv1d(out, filters = cfg.DIM_MODEL, kernel_size = 1)

    return out

def layer_norm(x):
    '''
    perform layer normalisation
    '''
    out = tf.contrib.layers.layer_norm(x, center = True, scale = True)
    return out

###### STACKS ######

def query_stack(q_in, mask, scope):
    '''
    Single query stack 
    Args:
        q_in:       (batch_size, seqlen, embed_size)
        input_mask: (batch_size, seqlen, seqlen)
    '''
    with tf.variable_scope(scope):
        out = layer_norm(out + multihead_attention(q_in, q_in, q_in, mask))
        out = layer_norm(out + feed_forward(out))

    return out

def passage_stack(p_in, q_out, input_mask, target_mask, scope):
    '''
    Single passage stack
    Args:
        p_in: (batch_size, seqlen, embed_size)
        q_out: output from query stack
    '''
    with tf.variable_scope(scope):
        out = layer_norm(p_in + multihead_attention(p_in, p_in, p_in,
                                                mask = target_mask,
                                                scope = 'self_attn'))
        out = layer_norm(out + multihead_attention(out, out, q_out,
                                                mask = input_mask))
        out = layer_norm(out + feed_forward(out))

    return out


###### REDUNDANT ######

def xx_attention(Q, K, V, mask):
    '''
    ###### REDUNDANT ######

    Attention Mechanism, formula is
    Att(Q,K,V) = softmax((Q*K')/sqrt(d_k))*V

    This returns a small operations in TF
    '''
    qkt = tf.matmul(Q, K, transpose_b = True)
    qkt /= np.swrt(cfg.DIM_KEY)
    if mask:
        qkt = qkt * mask
    soft_qkt = tf.nn.softmax(qkt)
    head = tf.matmul(soft_qkt, V)
    return head

def xx_multihead_attention(query, key, value, reuse, mask = None):
    '''
    ###### REDUNDANT ######

    code block to apply multi-head attention and return the
    '''
    scope = 'mha'
    if mask:
        scope = 'masked_mha'
    with tf.variable_scope(mha, reuse = reuse):
        head_tensors = [] # list stores all the output from head tensors
        for i in range(cfg.NUM_HEADS):
            # iterate over all the linear layers
            resuse_linear = True
            if not reuse:
                resuse_linear = False

            with tf.variable_scope('linear' + str(i), reuse = resuse_linear):
                # weights query
                W_q = tf.get_variable('W_QL' + str(i), shape = (cfg.DIM_MODEL, cfg.DIM_QUERY), 
                    initializer = cfg.LW_INITIALIZER)
                # weights key
                W_k = tf.get_variable('W_KL' + str(i), shape = (cfg.DIM_MODEL, cfg.DIM_KEY), 
                    initializer = cfg.LW_INITIALIZER)
                # weights value
                W_v = tf.get_variable('W_VL' + str(i), shape = (cfg.DIM_MODEL, cfg.DIM_VALUE), 
                    initializer = cfg.LW_INITIALIZER)

                # projections
                q_proj = tf.matmul(query, W_q)
                k_proj = tf.matmul(key, W_k)
                v_proj = tf.matmul(value, W_v)

                # perform attention and add to all the head layers
                if mask:
                    head_tensors.append(attention(q_proj, k_proj, v_proj, mask))
                else:
                    head_tensors.append(attention(q_proj, k_proj, v_proj))

        # concatenate and apply a single linear layer
        heads_concat = tf.reshape(tf.stack(head_tensors), [-1, cfg.DIM_MODEL])
        W_out = tf.get_variable('W_out_linear', shape = [heads_concat.shape[-1], cfg.DIM_MODEL])
        b_out = tf.get_variable('b_out_linear', shape = [1, cfg.DIM_MODEL])
        mha_out = tf.matmul(heads_concat, W_out) + b_out
        return mha_out
