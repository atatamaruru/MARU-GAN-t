import tensorflow as tf
from typing import List
from .common_layer import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization
from .embedding import AddPositionalEncoding
from .attention import MultiheadAttention, SelfAttention

"""BiGAN architecture.

Generator (decoder), encoder and discriminator.

"""

learning_rate = 0.00001
batch_size = 50
dis_inter_layer_dim = 100
init_kernel = tf.contrib.layers.xavier_initializer()
sequence_len = 12

hopping_num = 6
head_num = 8
hidden_dim = 512
dropout_rate = 0.1

variables_path = './variables.list'
with open(variables_path, 'w') as f:
    variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
    variable_names = [name for name in variable_names if not re.search('Adam', name)]
    f.write('\n'.join(variable_names) + '\n')

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]


def encoder(x_inp, is_training=False, getter=None, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        add_position_embedding = AddPositionalEncoding()
        input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        output_normalization = LayerNormalization()

        # [batch_size, length, hidden_dim]
        x_inp = tf.layers.dense(x_inp,
                                hidden_dim,
                                kernel_initializer=init_kernel)
        x_inp = tf.nn.relu(x_inp)
        #x_inp = tf.layers.dropout(x_inp, rate=0.1, name='dropout', training=is_training)
                       
        embedded_input = add_position_embedding(x_inp)
        query = input_dropout_layer(embedded_input, training=is_training)

        for i, layers in enumerate(attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = attention_layer(query, training=is_training)
                query = ffn_layer(query, training=is_training)
        
        # [batch_size, length, hidden_dim]
        print('enc')
        print(output_normalization(query).shape)
        return output_normalization(query)


def decoder(z_inp, is_training=False, getter=None, reuse=False):
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
        add_position_embedding = AddPositionalEncoding()
        input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        attention_block_list = []
        for _ in range(hopping_num):
            self_attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            enc_dec_attention_layer = MultiheadAttention(hidden_dim, head_num, dropout_rate, name='enc_dec_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            attention_block_list.append([
                ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate, name='enc_dec_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        output_normalization = LayerNormalization()
        output_dense_layer = tf.keras.layers.Dense(51, use_bias=False)

        # [batch_size, length, hidden_dim]
        initial_inp = tf.get_variable('x', shape=[1, hidden_dim])
        initial_inp = tf.tile(initial_inp, [batch_size, 1])
        initial_inp = tf.expand_dims(initial_inp, 1)
        
        print('initial_inp:{}'.format(initial_inp.shape))
        embedded_input = add_position_embedding(initial_inp)
        print('embedded_input:{}'.format(embedded_input.shape))
        query = input_dropout_layer(embedded_input, training=is_training)

        for i, layers in enumerate(attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = self_attention_layer(query, training=is_training)
                query = enc_dec_attention_layer(query, memory=z_inp, training=is_training)
                query = ffn_layer(query, training=is_training)

        query = output_normalization(query) #[batct_size, length, hidden_dim]
        print('gen_query')
        print(query.shape)
       
        query = output_dense_layer(query) # [batch_size, length, original_dim]
        #query = tf.layers.dense(tf.reshape(query, [batch_size*12, hidden_dim]),
        #                        51,
        #                        kernel_initializer=init_kernel)
        print('gen')
        print(z_inp.shape, query.shape)
        return z_inp, query#tf.reshape(query, [batch_size, 12, 51])


def discriminator(z_inp, x_inp, is_training=False, getter=None, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
        # D(x)
        name_x = 'x_layer_1'
        with tf.variable_scope(name_x):
            cell = tf.nn.rnn_cell.GRUCell(num_units=dis_inter_layer_dim) 
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 output_keep_prob=1.0-tf.to_float(is_training)*dropout_rate)
            outputs, _ = tf.nn.dynamic_rnn(cell, 
                                           x_inp,
                                           dtype=tf.float32)
            # print('outputs', outputs[:,-1,:]) #(?,100) (50,100)
        
        # D(z)
        name_z = 'z_layer_1'
        with tf.variable_scope(name_z):
            cell = tf.nn.rnn_cell.GRUCell(num_units=dis_inter_layer_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 output_keep_prob=1.0-tf.to_float(is_training)*dropout_rate)
            outputs_z, _ = tf.nn.dynamic_rnn(cell,
                                             z_inp,
                                             dtype=tf.float32)
 
        # D(x,z)
        # y = tf.concat([outputs[:,-1,:], z_inp], axis=-1)
        y = tf.concat([outputs[:,-1,:], outputs_z[:,-1,:]], axis=-1)                                        
        # print('y', y.shape) # (?, 200) (50,200)

        name_y = 'y_fc_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(y,
                                dis_inter_layer_dim,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, name='dropout', training=is_training)

        intermediate_layer = y

        name_y = 'y_fc_logits'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)
    # print('logitssize', logits.shape)
    return logits, intermediate_layer

def leakyReLu(x, alpha=0.1, name='leaky_relu'):
    """ Leaky relu """
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
