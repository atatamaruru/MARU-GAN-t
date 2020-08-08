import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import os
import re
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import bigan.SWaT_utilities_train_validtest_3layer as network
import data.dataset_train_validtest_swap as data

RANDOM_SEED = 51
FREQ_PRINT = 20 # print frequency image tensorboard [20]

def __init__(self):
    self.epoch = tf.get_variable('epoch', trainable=False, shape=[], dtype=tf.int32,
                                 initializer=tf.constant_initializer(0, dtype=tf.int32)) 
    self.add_epoch = tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32)))


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
       
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, weight, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Degree for L norms: ', degree)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, rd):
    ''' Directory to save training logs, weights, biases, etc.'''
    return 'bigan/train_logs/SWaT/'
    
def flatten(nested_list):
    '''2重のリストをフラットにする関数'''
    return [e for inner_list in nested_list for e in inner_list]

def train_and_test(nb_epochs, weight, method, degree, random_seed, date):
    ''' Runs the Bigan on the SWaT dataset
    '''
    logger = logging.getLogger('BiGAN.train.kdd.{}'.format(method))
    os.makedirs('./params/SWaT/{}'.format(date), exist_ok=True)
    os.makedirs('./results/SWaT/{}'.format(date), exist_ok=True)

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=(None, 12, 51), name='input') #None, sequence_length, dim of each data point
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name='lr_pl')

    # Data 
    trainx = data.get_train(sequence_len=12)
    trainx_copy = trainx.copy()
    # Use shuffle data
    validx, validy = data.get_valid(sequence_len=12)
    testx, testy  = data.get_test(sequence_len=12)
    
    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    #latent_dim = network.hidden_dim
    ema_decay = 0.9999
    sequence_len = network.sequence_len

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_valid = int(validx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)
    
    logger.info('Building training graph...')

    logger.warn('The BiGAN is training with the following parameters:')
    display_parameters(batch_size, starting_lr, ema_decay, weight, degree)

    gen = network.decoder
    enc = network.encoder
    dis = network.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, 512])
        z = tf.layers.dense(z,
                            512*12,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        z = tf.nn.relu(z)
        #z = tf.layers.dropout(z, rate=0.2, name='dropout', training=is_training)
        #print('z:{}'.format(z))
        z = tf.reshape(z, [batch_size, 12, 512])
        x_gen = gen(z, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl)
        l_generator, inter_layer_rct = dis(z, x_gen[1], is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc
        # generator
        loss_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator),logits=l_generator))
        # encoder
        loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder),logits=l_encoder))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)

        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema[1],
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema[1]
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                                keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == 'cross-e':
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_ema),logits=l_generator_ema)

            elif method == 'fm':
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                    keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score

    logdir = create_logdir(weight, method, random_seed)

    variables_path = './results/SWaT/{}/variables.list'.format(date)
    with open(variables_path, 'w') as f:
        variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
        variable_names = [name for name in variable_names if not re.search('Adam', name)]
        f.write('\n'.join(variable_names) + '\n')

    saver = tf.train.Saver()
    
    logger.info('Start training...')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0
        best_f1, best_epoch = 0, 0
        loss_train_gen, loss_train_enc, loss_train_dis = [], [], []

        #while not sv.should_stop() and epoch < nb_epochs:
        while epoch < nb_epochs:            
            lr = starting_lr
            begin = time.time()
            
            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0]

            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)             
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl:trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}

                _, ld, sm = sess.run([train_dis_op,
                                      loss_discriminator,
                                      sum_op_dis],
                                      feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator and encoder
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                
                _, _, le, lg, sm = sess.run([train_gen_op,
                                             train_enc_op,
                                             loss_encoder,
                                             loss_generator,
                                             sum_op_gen],
                                             feed_dict=feed_dict)
                
                train_loss_gen += lg
                train_loss_enc += le
                writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train

            loss_train_gen.append(train_loss_gen)
            loss_train_enc.append(train_loss_enc)
            loss_train_dis.append(train_loss_dis)
             
            logger.info('Epoch terminated')
            print('Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f'
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))           

            scores_valid = []
            latent_vars = np.empty((0, 12, 512))
            zs = np.empty((0, 12, 512))
            gdata = np.empty((0, 12, 51))
            # Create scores of valid data
            for t in range(nr_batches_valid):
                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                feed_dict = {input_pl: validx[ran_from:ran_to],
                             is_training_pl:False}
                             
                scores_valid += sess.run(list_scores,
                                         feed_dict=feed_dict).tolist()
            
                # get data of generator and encoder
                latent_var = sess.run(z_gen,
                                      feed_dict=feed_dict)
                latent_vars = np.append(latent_vars, latent_var, axis=0)
                z, generated_data = sess.run(x_gen,
                                             feed_dict=feed_dict)
                zs = np.append(zs, z, axis=0)
                print(validx[ran_from:ran_to])
                print(validx[ran_from:ran_to].shape)
                print(generated_data)
                print(generated_data.shape)
                gdata = np.append(gdata, generated_data, axis=0)
            
            np.save('./results/SWaT/{0}/enc_z_{1}'.format(date, epoch), latent_vars)
            np.save('./results/SWaT/{0}/gen_z_{1}'.format(date, epoch), zs)
            np.save('./results/SWaT/{0}/gen_x_{1}'.format(date, epoch), gdata)
            
            ran_from = nr_batches_valid * batch_size
            ran_to = (nr_batches_valid + 1) * batch_size
            size = validx[ran_from:ran_to].shape[0]
            fill = np.ones([batch_size - size, 12, 51])

            batch = np.concatenate([validx[ran_from:ran_to], fill], axis=0)
            feed_dict = {input_pl: batch,
                         is_training_pl: False}
            batch_score = sess.run(list_scores,
                               feed_dict=feed_dict).tolist()
            scores_valid += batch_score[:size]
            
            ratio = 20
            per = np.percentile(np.array(scores_valid), 100-ratio)
            
            y_pred = scores_valid.copy()
            y_pred = np.array(y_pred)

            inds = (y_pred < per)
            inds_comp = (y_pred >= per)

            y_pred[inds] = 0
            y_pred[inds_comp] = 1

            precision, recall, f1, _ = precision_recall_fscore_support(validy,
                                                                       y_pred,
                                                                       average='binary')
            accuracy = accuracy_score(validy, y_pred)
            
            print(
                'Validation : Prec = %.4f | Rec = %.4f | F1 = %.4f | Accuracy = %.4f'
                % (precision, recall, f1, accuracy))

            if f1 >= best_f1:
                best_f1 = f1
                best_epoch = epoch
                saver.save(sess, './params/SWaT/{}/model.ckpt-best'.format(date))
              
            print('best_f1 is {0} at {1} epoch)'.format(best_f1, best_epoch))
            print('-----------------------------')
            
            epoch += 1
      
        logger.warn('Testing evaluation...')

        saver.restore(sess, './params/SWaT/{}/model.ckpt-best'.format(date))
        scores_test = []
        inference_time = []
        #latent_vars = np.empty((0,300))        

        # Create scores of test data
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}
            scores_test += sess.run(list_scores,
                                    feed_dict=feed_dict).tolist()
            #latent_var = sess.run(z_gen,
            #                      feed_dict=feed_dict)
            #latent_var = np.array(latent_var).transpose((1,0,2)).reshape(50,300)
            #latent_vars = np.append(latent_vars, latent_var, axis=0)
            inference_time.append(time.time() - begin_val_batch)
        
        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))
        
        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 12, 51]) # sequence_len and input_dim

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl:batch,
                     is_training_pl:False}

        batch_score = sess.run(list_scores,
                               feed_dict=feed_dict).tolist()
        #batch_latent_var = sess.run(z_gen,
        #                            feed_dict=feed_dict)
        #batch_latent_var = np.array(batch_latent_var).transpose((1,0,2)).reshape(50,300)
                           
        scores_test += batch_score[:size]
        #batch_latent_var = batch_latent_var[0:size]
        #latent_vars = np.append(latent_vars, batch_latent_var, axis=0)

        per = np.percentile(np.array(scores_test), 100-ratio)

        y_pred = scores_test.copy()
        y_pred = np.array(y_pred)
        np.save('./results/SWaT/{}/AnomalyScores_test'.format(date), y_pred)

        inds = (y_pred < per)
        inds_comp = (y_pred >= per)
        
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        print('epoch = {0}'.format(best_epoch))
        print('anomalous = {0} | normal = {1} in prediction'.format(np.sum(y_pred == 1), np.sum(y_pred == 0)))      

        #print(testy)
        #print(y_pred)
        #print(np.array(latent_var).shape)
  
        precision, recall, f1,_ = precision_recall_fscore_support(testy,
                                                                  y_pred,
                                                                  average='binary')
        accuracy = accuracy_score(testy, y_pred)
       
        #tn, fp, fn, tp = confusion_matrix(testy, y_pred).ravel()
        #print('tp = {0} | fn = {1} | fp = {2} | tn = {3}'.format(tp, fn, fp, tn))
        print(
            'Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f | Accuracy = %.4f'
            % (precision, recall, f1, accuracy))
        """
        list_FP, list_FN, list_TP, list_TN = [], [], [], []

        for i in range(len(testx)):
            if testy[i] == 1:
                list_AN.append(i)
            if testy[i] == 0 and y_pred[i] == 1:
                list_FP.append(i)
            if testy[i] == 1 and y_pred[i] == 0:
                list_FN.append(i)
            if testy[i] == 0 and y_pred[i] == 0:
                list_TN.append(i)
            if testy[i] == 1 and y_pred[i] == 1:
                list_TP.append(i)

        np.save('./results/SWaT/{}/Latentvariables'.format(date), latent_vars)

        str_AN = '\n'.join(map(str,list_AN))
        with open('./results/SWaT/{}/NUM_AN.txt'.format(date), 'wt') as f:
            f.write(str_AN)
        str_FP = '\n'.join(map(str,list_FP))
        with open('./results/SWaT/{}/NUM_FP.txt'.format(date), 'wt') as f:
            f.write(str_FP)
        str_FN = '\n'.join(map(str,list_FN))
        with open('./results/SWaT/{}/NUM_FN.txt'.format(date), 'wt') as f:
            f.write(str_FN)
        str_TN = '\n'.join(map(str,list_TN))
        with open('./results/SWaT/{}/NUM_TN.txt'.format(date), 'wt') as f:
            f.write(str_TN)
        str_TP = '\n'.join(map(str,list_TP))
        with open('./results/SWaT/{}/NUM_TP.txt'.format(date), 'wt') as f:
            f.write(str_TP)
        """
def run(nb_epochs, weight, method, degree, random_seed, date):
    ''' Runs the training process'''
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed, date)
