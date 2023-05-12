#!/usr/bin/env python2

import sys
import tensorflow as tf
from flip_gradient import flip_gradient
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pickle
from DataHandler_doubling_only import ProcessData
import math
import sys
import scipy.io as sio
from matplotlib.pyplot import cm

config = tf.ConfigProto(device_count={'CPU': 1},
                        intra_op_parallelism_threads=2,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True)
session = tf.Session(config=config)

# running parameter
# USE_SHAPE_OF_SERIES = False
USE_VAE = False
PLOT_EMBEDDING = True
PLOT_PREDICTION = False
APPEND_TOXICITY_LABELS = False

NUMBER_OF_PREDICTION_PLOTS = 8

# Load data
data_handler = ProcessData()
data_handler.load_data_for_loo()


i_one = 0
# i_one = np.random.randint(0, 44)
print('uploaded data')
while True:

    
    print('LOOCV start')
    data, text = data_handler.leave_one_out(i_one=i_one, append_toxicity_labels=APPEND_TOXICITY_LABELS)
    print('LOOCV complete')
    if data is None:
        print('data is none')
        break
    i_one += 1

    target_x_train = data['TargetX_train']
    target_x_valid = data['TargetX_valid']
    source_x_train = data['SourceX_train']
    source_y_train = data['SourceY_train']
    source_x_valid = data['SourceX_valid']
    source_y_valid = data['SourceY_valid']
    tox_label_train = data['train_toxicity_labels']
    tox_label_valid = data['valid_toxicity_labels']
    

    short_labels_train = text['short_labels_train']
    short_labels_valid = text['short_labels_valid']
    all_usable_genes = text['gene_names']
    labels_valid = text['Labels_valid']
    labels_train = text['Labels_train']

    print()
    print(i_one)
    print(labels_train)
    print(labels_valid)
    print()
    # continue

    # Network parameter
    # NUMBER_OF_STEPS = 10000 #LLO
    NUMBER_OF_STEPS = 20000

    BATCH_SIZE = 128  # 0.0067

    HIDDN_NODED = [135, 94, 64, 120, 190]  # min pred loss

    INPUT_LAYER = source_x_train.shape[1]  # 151 

    VAE_HIDDEN_ENCODE_1 = HIDDN_NODED[0]  # 128
    VAE_HIDDEN_ENCODE_2 = HIDDN_NODED[1]  # 32
    LATENT_SPACE = HIDDN_NODED[2]
    VAE_HIDDEN_DECODE_1 = HIDDN_NODED[3]  # 128
    VAE_HIDDEN_DECODE_2 = HIDDN_NODED[4]  # 128

    OUTPUT_LAYER = source_y_train.shape[1]
    DOMAIN_HIDDEN_1 = 8
    DOMAIN_OUTPUT = 2

    # DOMAIN_LOSS_WEIGHT = 1.0

    PLOT_PREFIX = "Doubling_"

    # sys.exit()

    class UDAModel(object):
        """Simple unsupervised domain adaptation model."""

        def __init__(self):
            self._build_model()

        w_init = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
        # b_init = tf.zeros_initializer()
        b_init = tf.ones_initializer()

       
        @staticmethod
        def glorot_init(shape):
            # A custom initialization (see Xavier Glorot init)
            return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

        def vae_loss(self, x_reconstructed, x_true):
            # The loss is composed of two terms:
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            #     Adding 1e-10 to avoid evaluation of log(0.0)

            encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(
                1e-10 + 1 - x_reconstructed)
            encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)

            # 2.) The latent loss, which is defined as the Kullback Leibler divergence
            #     between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.
            #     But only for the source training instances. Otherwise matrix dimensions do not match

            all_z_std = lambda: self.z_std
            source_z_std = lambda: tf.slice(self.z_std, [0, 0], [BATCH_SIZE, -1])
            classify_z_std = tf.cond(self.train, source_z_std, all_z_std, name="cond_z_std")

            all_z_mean = lambda: self.z_mean
            source_z_mean = lambda: tf.slice(self.z_mean, [0, 0], [BATCH_SIZE, -1])
            classify_z_mean = tf.cond(self.train, source_z_mean, all_z_mean, name="cond_z_mean")

            kl_div_loss = 1 + classify_z_std - tf.square(classify_z_mean) - tf.exp(classify_z_std)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

            return tf.reduce_mean(encode_decode_loss + kl_div_loss)

        def _build_model(self):
            self.X = tf.placeholder(tf.float32, [None, INPUT_LAYER], name="input_x")  # input X
            self.y = tf.placeholder(tf.float32, [None, OUTPUT_LAYER], name="class_label_y")  # class label y
            self.domain = tf.placeholder(tf.float32, [None, DOMAIN_OUTPUT], name="domain_label_d")  # domain label d
            self.l = tf.placeholder(tf.float32, [], name="lambda")  # gradient reversal parameter (lambda)
            self.train = tf.placeholder(tf.bool, [], name="switch")  # switch for routing data to class predictor

            self.keep_prob = tf.placeholder_with_default(1.0, shape=())
            # MLP model for feature extraction
            with tf.variable_scope('in_vitro_encoder'):
                variables_ecd = {
                    'W_ecd0': tf.Variable(self.w_init(shape=[INPUT_LAYER, VAE_HIDDEN_ENCODE_1]), name="ecd_W_0"),
                    'b_ecd0': tf.Variable(self.b_init(shape=[VAE_HIDDEN_ENCODE_1]), name="ecd_b_0"),
                    'W_ecd0_5': tf.Variable(self.w_init(shape=[VAE_HIDDEN_ENCODE_1, VAE_HIDDEN_ENCODE_2]),
                                            name="ecd_W_0_5"),
                    'b_ecd0_5': tf.Variable(self.b_init(shape=[VAE_HIDDEN_ENCODE_2]), name="ecd_b_0_5"),

                    'W_z_mean': tf.Variable(self.w_init(shape=[VAE_HIDDEN_ENCODE_2, LATENT_SPACE]),
                                            name="ecd_W_z_mean"),
                    'b_z_mean': tf.Variable(self.b_init(shape=[LATENT_SPACE]), name="ecd_b_z_mean"),
   
                }

                encoder = tf.matmul(self.X, variables_ecd['W_ecd0']) + variables_ecd['b_ecd0']
                encoder = tf.nn.relu(encoder)
                self.encoder_1=encoder

                encoder = tf.matmul(encoder, variables_ecd['W_ecd0_5']) + variables_ecd['b_ecd0_5']
                encoder = tf.nn.relu(encoder)
                self.encoder_2=encoder



                self.z_mean = tf.matmul(encoder, variables_ecd['W_z_mean']) + variables_ecd['b_z_mean']
                self.latent_space = tf.nn.relu(self.z_mean)

                print('self.latent_space', self.latent_space)
   
            # MLP for class prediction
            with tf.variable_scope('in_vivo_decoder'):
                # Switches to route target examples (second half of batch) differently
                # depending on train or test mode.

                all_features = lambda: self.latent_space
                source_features = lambda: tf.slice(self.latent_space, [0, 0], [BATCH_SIZE, -1])
                classify_feats = tf.cond(self.train, source_features, all_features, name="cond_class_feat")

                target_features = tf.slice(self.latent_space, [BATCH_SIZE, 0], [BATCH_SIZE, -1])


                variables_fc = {
                    'W_fc0': tf.Variable(self.w_init(shape=[LATENT_SPACE, VAE_HIDDEN_DECODE_1]), name="fc_W_0"),
                    'b_fc0': tf.Variable(self.b_init(shape=[VAE_HIDDEN_DECODE_1]), name="fc_b_0"),
                    'W_fc0_5': tf.Variable(self.w_init(shape=[VAE_HIDDEN_DECODE_1, VAE_HIDDEN_DECODE_2]),
                                           name="fc_W_0_5"),
                    'b_fc0_5': tf.Variable(self.b_init(shape=[VAE_HIDDEN_DECODE_2]), name="fc_b_0_5"),
                    'W_fc2': tf.Variable(self.w_init(shape=[VAE_HIDDEN_DECODE_2, OUTPUT_LAYER]), name="fc_W_2"),
                    'b_fc2': tf.Variable(self.b_init(shape=[OUTPUT_LAYER]), name="fc_b_2")
                }

                decoder = tf.matmul(classify_feats, variables_fc['W_fc0']) + variables_fc['b_fc0']
                decoder = tf.nn.relu(decoder)
                decoder = tf.nn.dropout(decoder, keep_prob=self.keep_prob)
                decoder = tf.matmul(decoder, variables_fc['W_fc0_5']) + variables_fc['b_fc0_5']
                decoder = tf.nn.relu(decoder)
                decoder = tf.nn.dropout(decoder, keep_prob=self.keep_prob)
                decoder = tf.matmul(decoder, variables_fc['W_fc2']) + variables_fc['b_fc2']
                self.pred = tf.nn.sigmoid(decoder)

                target_var = tf.matmul(target_features, variables_fc['W_fc0']) + variables_fc['b_fc0']
                target_var = tf.nn.relu(target_var)
                target_var = tf.nn.dropout(target_var, keep_prob=self.keep_prob)
                target_var = tf.matmul(target_var, variables_fc['W_fc0_5']) + variables_fc['b_fc0_5']
                target_var = tf.nn.relu(target_var)
                target_var = tf.nn.dropout(target_var, keep_prob=self.keep_prob)
                target_var = tf.matmul(target_var, variables_fc['W_fc2']) + variables_fc['b_fc2']
                self.target_variance = tf.nn.moments(tf.nn.sigmoid(target_var), [0])[1]

                if USE_VAE:
                    # loss VAE
                    self.pred_loss = self.vae_loss(self.pred, self.y)
                else:
                    # loss MAE
                    weights = np.ones(OUTPUT_LAYER, dtype='float32')
                    if APPEND_TOXICITY_LABELS:
                        weights[0:OUTPUT_LAYER - 2:4] = 0.8
                    else:
                        weights[0::4] = 0.8
                    tf_weights = tf.constant(weights)

                    print(tf_weights)
                    self.pred_loss = tf.reduce_sum(tf.abs(self.y - self.pred)) # mse 0.0096


                    # sys.exit()



                print('self.pred_loss', self.pred_loss)

            # Small MLP for domain prediction with adversarial loss
            with tf.variable_scope('domain_decoder'):
                # Flip the gradient when backpropagating through this operation

                feat = flip_gradient(self.latent_space, self.l)

                variables_dp = {
                    'W_dp0': tf.Variable(self.w_init(shape=[LATENT_SPACE, DOMAIN_HIDDEN_1]), name="dp_W_0"),
                    'b_dp0': tf.Variable(self.b_init(shape=[DOMAIN_HIDDEN_1]), name="dp_b_0"),
                    'W_dp1': tf.Variable(self.w_init(shape=[DOMAIN_HIDDEN_1, DOMAIN_OUTPUT]), name="dp_W_1"),
                    'b_dp1': tf.Variable(self.b_init(shape=[DOMAIN_OUTPUT]), name="dp_b_1"),
                }
                d_h_fc0 = tf.nn.relu(tf.matmul(feat, variables_dp['W_dp0']) + variables_dp['b_dp0'])
                d_logits = tf.matmul(d_h_fc0, variables_dp['W_dp1']) + variables_dp['b_dp1']

                self.domain_pred = tf.nn.softmax(d_logits)

                
                self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)


    # Build the model graph
    graph = tf.get_default_graph()
    with graph.as_default():
        model = UDAModel()

        learning_rate = tf.placeholder(tf.float32, [])

        pred_variance = tf.reduce_mean(tf.nn.moments(model.pred, [0])[1])

        target_variance = tf.reduce_mean(model.target_variance)

        pred_loss = tf.reduce_mean(model.pred_loss)
        domain_loss = tf.reduce_mean(model.domain_loss)


        total_loss = pred_loss 
        
        print('cost function :', total_loss)


        cost_fn = total_loss
        if USE_VAE:
            dann_train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost_fn)
        else:
            dann_train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost_fn)
 

        print('dann_train_op', dann_train_op)
        # Evaluation
        pred_mse = tf.reduce_mean(tf.squared_difference(model.pred, model.y))
        domain_mse = tf.reduce_mean(tf.squared_difference(model.domain_pred, model.domain))
        correct_domain_predic = tf.equal(tf.argmax(model.domain,1),tf.argmax(model.domain_pred,1))
        pred_mae = tf.reduce_mean(tf.abs(model.y - model.pred))
        domain_mae = tf.reduce_mean(tf.abs(model.domain - model.domain_pred))


    def train_and_evaluate(graph, model, num_steps=NUMBER_OF_STEPS, verbose=True):
        """Helper to run the model with different training modes."""
        
        with tf.Session(graph=graph) as sess:

            tf.global_variables_initializer().run()
            # tf.local_variables_initializer().run()
            


            domain_labels = np.vstack([np.tile([1., 0.], [BATCH_SIZE, 1]),
                                       np.tile([0., 1.], [BATCH_SIZE, 1])])

            all_domain_labels = np.vstack([np.tile([1., 0.], [source_x_train.shape[0], 1]),
                                           np.tile([0., 1.], [target_x_train.shape[0], 1])])

            # Training loop
            for i in range(num_steps):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                # l = p ** 2
                # lr = 0.01 / (1. + 10 * p) ** 0.75
                lr = 0.0001 / (1. + 10 * p) ** 0.75

                shuffle_indices = np.random.permutation(np.arange(len(source_y_train)))
                source_x_train_shuffled = source_x_train[shuffle_indices]
                source_y_train_shuffled = source_y_train[shuffle_indices]
                target_x_train_shuffled = target_x_train[shuffle_indices]

                # Minibatch training
                for b in range(0, len(source_y_train_shuffled) // BATCH_SIZE):
                    start = b * BATCH_SIZE
                    Xs = source_x_train_shuffled[start:start + BATCH_SIZE]
                    ys = source_y_train_shuffled[start:start + BATCH_SIZE]
                    Xt = target_x_train_shuffled[start:start + BATCH_SIZE]
                    X = np.vstack([Xs, Xt])

                    # Run optimizer with batch
                    _, batch_loss = sess.run([dann_train_op, total_loss],
                                             feed_dict={model.X: X, model.y: ys, model.domain: domain_labels,
                                                        model.train: True, model.l: l, learning_rate: lr,
                                                        model.keep_prob: 0.75})

                if verbose and p*100 in[0,25,50,75]: 
                    print(p*100)
                    source_train_compound_mse = sess.run(pred_mse,
                                                            feed_dict={model.X: source_x_train, model.y: source_y_train,
                                                                        model.train: False})

                    source_train_domain_mse = sess.run(domain_mse,
                                                        feed_dict={model.X: source_x_train,
                                                                    model.domain: np.tile([1., 0.], [len(source_x_train), 1]),
                                                                    model.train: False})
        
        
                    s_train_domain_acc = sess.run(correct_domain_predic,
                                                        feed_dict={model.X: source_x_train,
                                                                    model.domain: np.tile([1., 0.], [len(source_x_train), 1]),
                                                                    model.train: False})
        
                    source_train_domain_acc = sum(s_train_domain_acc == True)/len(s_train_domain_acc)
    
                    target_train_domain_mse = sess.run(domain_mse,
                                                        feed_dict={model.X: target_x_train,
                                                                    model.domain: np.tile([0., 1.], [len(target_x_train), 1]),
                                                                    model.train: False})
        
                    t_train_domain_acc = sess.run(correct_domain_predic,
                                                        feed_dict={model.X: target_x_train,
                                                                    model.domain: np.tile([0., 1.], [len(target_x_train), 1]),
                                                                    model.train: False})
        
                    target_train_domain_acc = sum(t_train_domain_acc == True)/len(t_train_domain_acc)
                    t_mse = {
                        'source_compound': source_train_compound_mse,
                        'source_domain': source_train_domain_mse,
                        'source_domain_acc': source_train_domain_acc,
                        'target_domain': target_train_domain_mse,
                        'target_domain_acc': target_train_domain_acc
                    }

                    source_valid_compound_mse = sess.run(pred_mse,
                                                            feed_dict={model.X: source_x_valid, model.y: source_y_valid,
                                                                    model.train: False})

                    source_valid_domain_mse = sess.run(domain_mse,
                                                        feed_dict={model.X: source_x_valid,
                                                                    model.domain: np.tile([1., 0.], [len(source_x_valid), 1]),
                                                                    model.train: False})
                    s_valid_domain_acc = sess.run(correct_domain_predic,
                                                        feed_dict={model.X: source_x_valid,
                                                                    model.domain: np.tile([1., 0.], [len(source_x_valid), 1]),
                                                                    model.train: False})
        
                    source_valid_domain_acc = sum(s_valid_domain_acc == True)/len(s_valid_domain_acc)

                    target_valid_domain_mse = sess.run(domain_mse,
                                                        feed_dict={model.X: target_x_valid,
                                                                    model.domain: np.tile([0., 1.], [len(target_x_valid), 1]),
                                                                    model.train: False})
        
                    t_valid_domain_acc = sess.run(correct_domain_predic,
                                                        feed_dict={model.X: target_x_valid,
                                                                    model.domain: np.tile([0., 1.], [len(target_x_valid), 1]),
                                                                    model.train: False})
            
                    target_valid_domain_acc = sum(t_valid_domain_acc==True)/len(t_valid_domain_acc)
      

                    v_mse = {
                        'source_compound': source_valid_compound_mse,
                        'source_domain': source_valid_domain_mse,
                        'source_domain_acc': source_valid_domain_acc,
                        'target_domain': target_valid_domain_mse,
                        'target_domain_acc': target_valid_domain_acc
                    }
        
                    source_train_emb = sess.run(model.encoder_1, feed_dict={model.X: source_x_train})
                    target_train_emb = sess.run(model.encoder_1, feed_dict={model.X: target_x_train})
                    source_valid_emb = sess.run(model.encoder_1, feed_dict={model.X: source_x_valid})
                    target_valid_emb = sess.run(model.encoder_1, feed_dict={model.X: target_x_valid})
                    embed_encoder1 = {
                        'source_train': source_train_emb,
                        'target_train': target_train_emb,
                        'source_valid': source_valid_emb,
                        'target_valid': target_valid_emb
                    }
        
                    source_train_emb = sess.run(model.encoder_2, feed_dict={model.X: source_x_train})
                    target_train_emb = sess.run(model.encoder_2, feed_dict={model.X: target_x_train})
                    source_valid_emb = sess.run(model.encoder_2, feed_dict={model.X: source_x_valid})
                    target_valid_emb = sess.run(model.encoder_2, feed_dict={model.X: target_x_valid})
                    embed_encoder2= {
                        'source_train': source_train_emb,
                        'target_train': target_train_emb,
                        'source_valid': source_valid_emb,
                        'target_valid': target_valid_emb
                    }
        
        
                    source_train_emb = sess.run(model.latent_space, feed_dict={model.X: source_x_train})
                    target_train_emb = sess.run(model.latent_space, feed_dict={model.X: target_x_train})
                    source_valid_emb = sess.run(model.latent_space, feed_dict={model.X: source_x_valid})
                    target_valid_emb = sess.run(model.latent_space, feed_dict={model.X: target_x_valid})
                    embed_latent = {
                        'source_train': source_train_emb,
                        'target_train': target_train_emb,
                        'source_valid': source_valid_emb,
                        'target_valid': target_valid_emb
                    }

                    source_train_pred = sess.run(model.pred, feed_dict={model.X: source_x_train, model.train: False})
                    target_train_pred = sess.run(model.pred, feed_dict={model.X: target_x_train, model.train: False})
                    source_valid_pred = sess.run(model.pred, feed_dict={model.X: source_x_valid, model.train: False})
                    target_valid_pred = sess.run(model.pred, feed_dict={model.X: target_x_valid, model.train: False})
                    predict = {
                        'source_train': data_handler.correct_data(source_train_pred, APPEND_TOXICITY_LABELS),
                        'target_train': data_handler.correct_data(target_train_pred, APPEND_TOXICITY_LABELS, target=True),
                        'source_valid': data_handler.correct_data(source_valid_pred, APPEND_TOXICITY_LABELS),
                        'target_valid': data_handler.correct_data(target_valid_pred, APPEND_TOXICITY_LABELS, target=True)
                    }
        
        
        
            
                    #rnd_train_indices = range(len(embed_latent['source_train']))
                    #
                    #rnd_valid_indices = range(len(embed_latent['source_valid']))
                   

        
                    with open("midterm/STEATOSIS_pred_720LE_L1O_matched/interval_"+ str(p*100) + '_' + str(i_one) + ".p", "wb") as outfile:
                        pickle.dump((t_mse,
                                    v_mse,
                                    embed_encoder1,
                                    embed_encoder2,
                                    embed_latent,
                                    tox_label_train,
                                    tox_label_valid), outfile)
                
                
                if verbose and i % 100 == 0:
                    x_complete = np.vstack([source_x_train_shuffled, target_x_train_shuffled])
                    dloss, d_mse,d_mae = sess.run(
                        [domain_loss, domain_mse, domain_mae],
                        feed_dict={model.X: x_complete, model.domain: all_domain_labels, model.train: False})

                    ploss, p_mse, p_mae, batch_variance, t_variance = sess.run(
                        [pred_loss, pred_mse, pred_mae, pred_variance, target_variance],
                        feed_dict={model.X: source_x_train_shuffled, model.y: source_y_train_shuffled,
                                   model.train: False})

                    print(
                        '{:.1f}% - total loss: {:.4f} p_loss: {:.4f} p_MSE: {:.4f}  p_MAE {:.4f} d_loss: {:.4f} d_MSE: {:.4f} d_MAE{:.4f} l: {:.4f}  lr: {:.4f}  Svar: {:.4f}  Tvar: {:.4f}'.format(
                            p * 100, batch_loss, ploss, p_mse, p_mae, dloss, d_mse, d_mae, l, lr, batch_variance, t_variance))

                    
            # Compute final evaluation on test data

            source_train_compound_mse = sess.run(pred_mse,
                                                 feed_dict={model.X: source_x_train, model.y: source_y_train,
                                                            model.train: False})

            source_train_domain_mse = sess.run(domain_mse,
                                               feed_dict={model.X: source_x_train,
                                                          model.domain: np.tile([1., 0.], [len(source_x_train), 1]),
                                                          model.train: False})
            source_train_compound_mae = sess.run(pred_mae,
                                                 feed_dict={model.X: source_x_train, model.y: source_y_train,
                                                            model.train: False})

   
            s_train_domain_acc = sess.run(correct_domain_predic,
                                            feed_dict={model.X: source_x_train,
                                                        model.domain: np.tile([1., 0.], [len(source_x_train), 1]),
                                                        model.train: False})
        
            source_train_domain_acc = sum(s_train_domain_acc == True)/len(s_train_domain_acc)
            

            target_train_domain_mse = sess.run(domain_mse,
                                               feed_dict={model.X: target_x_train,
                                                          model.domain: np.tile([0., 1.], [len(target_x_train), 1]),
                                                          model.train: False})

                                                          
            
            t_train_domain_acc = sess.run(correct_domain_predic,
                                            feed_dict={model.X: target_x_train,
                                                        model.domain: np.tile([0., 1.], [len(target_x_train), 1]),
                                                        model.train: False})
        
            target_train_domain_acc = sum(t_train_domain_acc == True)/len(t_train_domain_acc)
            t_mse = {
                'source_compound': source_train_compound_mse,
                'source_compound_mae': source_train_compound_mae,
                'source_domain': source_train_domain_mse,
                'source_domain_acc': source_train_domain_acc,
                'target_domain': target_train_domain_mse,
                'target_domain_acc': target_train_domain_acc
            }

            source_valid_compound_mse = sess.run(pred_mse,
                                                 feed_dict={model.X: source_x_valid, model.y: source_y_valid,
                                                            model.train: False})

            source_valid_domain_mse = sess.run(domain_mse,
                                               feed_dict={model.X: source_x_valid,
                                                          model.domain: np.tile([1., 0.], [len(source_x_valid), 1]),
                                                          model.train: False})
            source_valid_compound_mae = sess.run(pred_mae,
                                                 feed_dict={model.X: source_x_valid, model.y: source_y_valid,
                                                            model.train: False})
            
            s_valid_domain_acc = sess.run(correct_domain_predic,
                                            feed_dict={model.X: source_x_valid,
                                                        model.domain: np.tile([1., 0.], [len(source_x_valid), 1]),
                                                        model.train: False})
        
            source_valid_domain_acc = sum(s_valid_domain_acc == True)/len(s_valid_domain_acc)

            target_valid_domain_mse = sess.run(domain_mse,
                                               feed_dict={model.X: target_x_valid,
                                                          model.domain: np.tile([0., 1.], [len(target_x_valid), 1]),
                                                          model.train: False})

            t_valid_domain_acc = sess.run(correct_domain_predic,
                                            feed_dict={model.X: target_x_valid,
                                                        model.domain: np.tile([0., 1.], [len(target_x_valid), 1]),
                                                        model.train: False})
            
            target_valid_domain_acc = sum(t_valid_domain_acc==True)/len(t_valid_domain_acc)
      

            v_mse = {
                'source_compound': source_valid_compound_mse,
                'source_compound_mae': source_valid_compound_mae,
                'source_domain': source_valid_domain_mse,
                'source_domain_acc': source_valid_domain_acc,
                'target_domain': target_valid_domain_mse,
                'target_domain_acc': target_valid_domain_acc
            }
        

            source_train_emb = sess.run(model.encoder_1, feed_dict={model.X: source_x_train})
            target_train_emb = sess.run(model.encoder_1, feed_dict={model.X: target_x_train})
            source_valid_emb = sess.run(model.encoder_1, feed_dict={model.X: source_x_valid})
            target_valid_emb = sess.run(model.encoder_1, feed_dict={model.X: target_x_valid})
            embed_encoder1 = {
            'source_train': source_train_emb,
            'target_train': target_train_emb,
            'source_valid': source_valid_emb,
            'target_valid': target_valid_emb
            }
        
            source_train_emb = sess.run(model.encoder_2, feed_dict={model.X: source_x_train})
            target_train_emb = sess.run(model.encoder_2, feed_dict={model.X: target_x_train})
            source_valid_emb = sess.run(model.encoder_2, feed_dict={model.X: source_x_valid})
            target_valid_emb = sess.run(model.encoder_2, feed_dict={model.X: target_x_valid})
            embed_encoder2= {
                'source_train': source_train_emb,
                'target_train': target_train_emb,
                'source_valid': source_valid_emb,
                'target_valid': target_valid_emb
            }

            source_train_emb = sess.run(model.latent_space, feed_dict={model.X: source_x_train})
            target_train_emb = sess.run(model.latent_space, feed_dict={model.X: target_x_train})
            source_valid_emb = sess.run(model.latent_space, feed_dict={model.X: source_x_valid})
            target_valid_emb = sess.run(model.latent_space, feed_dict={model.X: target_x_valid})
            embed_latent = {
                'source_train': source_train_emb,
                'target_train': target_train_emb,
                'source_valid': source_valid_emb,
                'target_valid': target_valid_emb
            }
        
            source_train_pred = sess.run(model.pred, feed_dict={model.X: source_x_train, model.train: False})
            target_train_pred = sess.run(model.pred, feed_dict={model.X: target_x_train, model.train: False})
            source_valid_pred = sess.run(model.pred, feed_dict={model.X: source_x_valid, model.train: False})
            target_valid_pred = sess.run(model.pred, feed_dict={model.X: target_x_valid, model.train: False})
            predict = {
                'source_train': data_handler.correct_data(source_train_pred, APPEND_TOXICITY_LABELS),
                'target_train': data_handler.correct_data(target_train_pred, APPEND_TOXICITY_LABELS, target=True),
                'source_valid': data_handler.correct_data(source_valid_pred, APPEND_TOXICITY_LABELS),
                'target_valid': data_handler.correct_data(target_valid_pred, APPEND_TOXICITY_LABELS, target=True)
            }

        return t_mse, v_mse, embed_encoder1, embed_encoder2,embed_latent, predict


    print('\nDomain adaptation training')

    training_mse, validation_mse, embeddings_e1, embeddings_e2, embeddings_latent, predictions = train_and_evaluate(graph, model)
 

    # pickle.dump([training_mse, validation_mse, embeddings, predictions], open("export/predictions_lables_loo_2"+str(i_one)+".p", "wb"))

    print('Source Training (Rat) Compound MSE:', training_mse['source_compound'])
    print('Source Training (Rat) Compound MAE:', training_mse['source_compound_mae'])
    print('Source Training (Rat) Domain MSE:', training_mse['source_domain'])
    print('Source Training (Rat) Domain accuracy', training_mse['source_domain_acc'])
    print('Target Training (Human) Domain MSE:', training_mse['target_domain'])
    print('Target Training (Human) Domain accuracy', training_mse['target_domain_acc'])
    print('Source Validate (Rat) Compound MSE:', validation_mse['source_compound'])
    print('Source Validate (Rat) Compound MAE:', validation_mse['source_compound_mae'])
    print('Source Validate (Rat) Domain MSE:', validation_mse['source_domain'])
    print('Source Validate (Rat) Domain accuracy', validation_mse['source_domain_acc'])
    print('Target Validate (Human) Domain MSE:', validation_mse['target_domain'])
    print('Target Validate (Human) Domain accuracy', validation_mse['target_domain_acc'])
    print('')

    print(len(embeddings_latent['source_train']))
    rnd_train_indices = range(len(embeddings_latent['source_train']))
    rnd_valid_indices = range(len(embeddings_latent['source_valid']))

    with open("export/STEATOSIS_pred_720LE_L1O_matched/predictions_" + str(i_one) + ".p", "wb") as outfile:
        pickle.dump((data_handler.get_source_x_train_original(),
                     data_handler.get_target_x_train_original(),
                     data_handler.get_source_x_valid_original(),
                     data_handler.get_target_x_valid_original(),
                     data_handler.get_source_y_train_original(),
                     data_handler.get_source_y_valid_original(),
                     training_mse,
                     validation_mse,
                     embeddings_e1,
                     embeddings_e2,
                     embeddings_latent,
                     predictions,
                     all_usable_genes,
                     labels_train,
                     labels_valid,
                     tox_label_train,
                     tox_label_valid), outfile)
        
   
                     

    def plot_embedding(X, labels, colour_id, name):
        """Plot an embedding X with the class label y colored by the domain d."""

        cmap = cm.tab20c(np.linspace(0, 1, 20))
        colours = [cmap[0], cmap[2], cmap[3],cmap[4], cmap[6],cmap[7], cmap[8], cmap[9],cmap[12],cmap[13]]


        for i in range(X.shape[0]):
            # plot colored number
            plt.text(X[i, 0], X[i, 1], labels[i],
                     color=colours[colour_id],
                     fontdict={'weight': 'bold', 'size': 9})
            plt.scatter(X[i, 0], X[i, 1],
                        color=colours[colour_id])
            #print(X[i])
        plt.plot([], [], color=colours[colour_id], label=name)


    if PLOT_EMBEDDING:
        embeddings=embeddings_latent


        if embeddings['source_train'][rnd_train_indices].shape[1] == 2:
            source_train_tsne = embeddings['source_train'][rnd_train_indices]
            target_train_tsne = embeddings['target_train'][rnd_train_indices]
            source_valid_tsne = embeddings['source_valid'][rnd_valid_indices]
            target_valid_tsne = embeddings['target_valid'][rnd_valid_indices]
        else:
            
            embedding_combi = np.concatenate((embeddings['source_train'][rnd_train_indices],embeddings['target_train'][rnd_train_indices],embeddings['source_valid'][rnd_valid_indices],embeddings['target_valid'][rnd_valid_indices]))
                                     
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)  # 3000
            print("TSNE:")
            #source_train_tsne = tsne.fit_transform(embeddings['source_train'][rnd_train_indices])
            #print("\t25%")
            #target_train_tsne = tsne.fit_transform(embeddings['target_train'][rnd_train_indices])
            #print("\t50%")
            #source_valid_tsne = tsne.fit_transform(embeddings['source_valid'][rnd_valid_indices])
            #print("\t75%")
            #target_valid_tsne = tsne.fit_transform(embeddings['target_valid'][rnd_valid_indices])
            #print("\t100%\n")
            
            combi_tsne = tsne.fit_transform(embedding_combi)
            
            
            #out=combi_tsne.copy()
            #source_train_tsne = out[0:1408,:]
            #target_train_tsne = out[1408:2816,:]
            #source_valid_tsne = out[2816:2848,:]
            #target_valid_tsne = out[2848:2880,:]
            
            #
            #out=combi_tsne.copy()
            #source_train_tsne = out[0:328,:]
            #target_train_tsne = out[328:656,:]
            #source_valid_tsne = out[656:688,:]
            #target_valid_tsne = out[688:720,:]
            
            out=combi_tsne.copy()
            source_train_tsne = out[0:704,:]
            target_train_tsne = out[704:1408,:]
            source_valid_tsne = out[1408:1424,:]
            target_valid_tsne = out[1424:1440,:]



        

        plt.figure(figsize=(10, 10))
        # ax = plt.subplot(111)
        plot_embedding(source_train_tsne, short_labels_train[rnd_train_indices], 0, 'Rat')
        plot_embedding(target_train_tsne, short_labels_train[rnd_train_indices], 3, 'Human')
        #plt.xticks([]), plt.yticks([])
        plt.title('Domain Adaptation Train', fontsize=16)

        #plt.xlim((-.05, 1.1))
        #plt.ylim((-.05, 1.05))
        plt.legend(prop={'size': 16})

        plt.savefig('plots/STEATOSIS_pred_720LE_L1O_matched/' + PLOT_PREFIX + 'latentTrain' + str(i_one))
        

        plt.figure(figsize=(10, 10))
        plot_embedding(source_train_tsne, short_labels_train[rnd_train_indices], 0, 'Rat train')
        plot_embedding(target_train_tsne, short_labels_train[rnd_train_indices], 3, 'Human train')
        plot_embedding(source_valid_tsne, short_labels_valid[rnd_valid_indices], 1, 'Rat valid')
        plot_embedding(target_valid_tsne, short_labels_valid[rnd_valid_indices], 4, 'Human valid')
        
        

        #plt.xticks([]), plt.yticks([])
        plt.title('Domain Adaptation Validate', fontsize=16)
        #plt.xlim((-.05, 1.1))
        #plt.ylim((-.05, 1.05))
        plt.legend(prop={'size': 16})
        #plt.savefig('plots/GTX_C_pred_dom/' + PLOT_PREFIX + 'latentValidate'+ str(i_one)+'.eps', format='eps',dpi=400)
        plt.savefig('plots/STEATOSIS_pred_720LE_L1O_matched/' + PLOT_PREFIX + 'latentValidate'+ str(i_one))
        
        X = source_train_tsne.copy()
        X_lab = short_labels_train[rnd_train_indices].copy()
        Y = target_train_tsne.copy()
        Y_lab = short_labels_train[rnd_train_indices].copy()
        
        X_GTX=X[tox_label_train[:,0]==1]
        X_GTX_lab=X_lab[tox_label_train[:,0]==1]
        
        X_NGTX=X[tox_label_train[:,0]==0]
        X_NGTX_lab=X_lab[tox_label_train[:,0]==0]
        
        Y_GTX=Y[tox_label_train[:,0]==1]
        Y_GTX_lab=Y_lab[tox_label_train[:,0]==1]
        
        Y_NGTX=Y[tox_label_train[:,0]==0]
        Y_NGTX_lab=Y_lab[tox_label_train[:,0]==0]
        
        plt.figure(figsize=(10, 10))
        plot_embedding(source_train_tsne, short_labels_train[rnd_train_indices], 2, 'Rat')
        plot_embedding(target_train_tsne, short_labels_train[rnd_train_indices], 5, 'Human')
        plot_embedding(X_GTX,X_GTX_lab,6,'rat_GTX')
        plot_embedding(X_NGTX,X_NGTX_lab,8,'rat_NGTX')
        plot_embedding(Y_GTX,Y_GTX_lab,7,'human_GTX')
        plot_embedding(Y_NGTX,Y_NGTX_lab,9,'human_NGTX')
        
        #plt.xticks([]), plt.yticks([])
        plt.title('Domain Adaptation Validate - GTX', fontsize=16)
        #plt.xlim((-.05, 1.1))
        #plt.ylim((-.05, 1.05))
        plt.legend(prop={'size': 16})
        #plt.savefig('plots/GTX_C_pred_dom/' + PLOT_PREFIX + 'latent_GTX_'+ str(i_one) +'.eps', format='eps',dpi=400)
        plt.savefig('plots/STEATOSIS_pred_720LE_L1O_matched/' + PLOT_PREFIX + 'latent_GTX_'+ str(i_one))
        
        
        
        
        X_C=X[tox_label_train[:,1]==1]
        X_C_lab=X_lab[tox_label_train[:,1]==1]
        
        X_NC=X[tox_label_train[:,1]==0]
        X_NC_lab=X_lab[tox_label_train[:,1]==0]
        
        Y_C=Y[tox_label_train[:,1]==1]
        Y_C_lab=Y_lab[tox_label_train[:,1]==1]
        
        Y_NC=Y[tox_label_train[:,1]==0]
        Y_NC_lab=Y_lab[tox_label_train[:,1]==0]
       
        plt.figure(figsize=(10, 10))
        plot_embedding(source_train_tsne, short_labels_train[rnd_train_indices], 2, 'Rat')
        plot_embedding(target_train_tsne, short_labels_train[rnd_train_indices], 5, 'Human')
        plot_embedding(X_C,X_C_lab,6,'rat_C')
        plot_embedding(X_NC,X_NC_lab,8,'rat_NC')
        plot_embedding(Y_C,Y_C_lab,7,'human_C')
        plot_embedding(Y_NC,Y_NC_lab,9,'human_NC')
        
        #plt.xticks([]), plt.yticks([])
        plt.title('Domain Adaptation Validate - CAR', fontsize=16)
        #plt.xlim((-.05, 1.1))
        #plt.ylim((-.05, 1.05))
        plt.legend(prop={'size': 16})
        #plt.savefig('plots/GTX_C_pred_dom/' + PLOT_PREFIX + 'latent_C_'+ str(i_one) +'.eps', format='eps',dpi=400)
        plt.savefig('plots/STEATOSIS_pred_720LE_L1O_matched/' + PLOT_PREFIX + 'latent_C_'+ str(i_one))
        
    

        

        

        plt.show()


    def prediction_plot_nones(source_x, source_y, target_x, ind, predict_source, predict_target, title, labels):
        plt.figure(figsize=[15, 10])
        plt.subplot(211)
        # numberGenes = int(source_x.shape[1] / 3)
        # for i in range(numberGenes):
        plt.plot(source_x[ind], color='b')
        # numberGenes = int(source_y.shape[1] / 4)
        # for i in range(numberGenes):
        plt.plot(source_y[ind], color='r')
        plt.plot(predict_source[ind], color='g')

        plt.plot([], [], color='b', label='Rat in-vitro measured')
        plt.plot([], [], color='r', label='Rat in-vivo measured')
        plt.plot([], [], color='g', label='Rat in-vivo predicted')
        plt.xticks([i * 5 for i in range(len(all_usable_genes))], all_usable_genes, rotation='vertical', fontsize=14)
        plt.legend(loc='upper left', prop={'size': 16})
        plt.title(title + ' ' + labels[ind], fontsize=16)
        plt.xlim((-1, source_y.shape[1]))
        plt.tight_layout()

        plt.subplot(212)
        # numberGenes = int(target_x.shape[1] / 3)
        # for i in range(numberGenes):
        plt.plot(target_x[ind], color='b')
        # numberGenes = int(predict_target.shape[1] / 4)
        # for i in range(numberGenes):
        plt.plot(predict_target[ind], color='g')

        plt.plot([], [], color='b', label='Human in-vitro measured')
        plt.plot([], [], color='g', label='Human in-vivo predicted')
        plt.xticks([i * 5 for i in range(len(all_usable_genes))], all_usable_genes, rotation='vertical', fontsize=14)
        plt.legend(loc='upper left', prop={'size': 16})
        # plt.title('Training Target (Human) ' + labels_train[ind])
        plt.xlim((-1, source_y.shape[1]))
        plt.tight_layout()


    if PLOT_PREDICTION:
 
        if APPEND_TOXICITY_LABELS:
            source_y_train_nones = np.full((data_handler.get_source_y_train_original().shape[0],
                                            data_handler.get_source_y_train_original().shape[1] + int(
                                                data_handler.get_source_y_train_original().shape[1] / 4)), None)
            source_x_train_nones = np.full((source_y_train_nones.shape[0], source_y_train_nones.shape[1] - 2), None)
            target_x_train_nones = np.full((source_y_train_nones.shape[0], source_y_train_nones.shape[1] - 2), None)


        else:
            source_y_train_nones = np.full((data_handler.get_source_y_train_original().shape[0],
                                            data_handler.get_source_y_train_original().shape[1] + int(
                                                data_handler.get_source_y_train_original().shape[1] / 4) - 1), None)
            source_x_train_nones = np.full(source_y_train_nones.shape, None)
            target_x_train_nones = np.full(source_y_train_nones.shape, None)

        source_pred_train_nones = np.full(source_y_train_nones.shape, None)
        target_pred_train_nones = np.full(source_y_train_nones.shape, None)

        if APPEND_TOXICITY_LABELS:
            source_y_valid_nones = np.full((data_handler.get_source_y_valid_original().shape[0],
                                            data_handler.get_source_y_valid_original().shape[1] + int(
                                                data_handler.get_source_y_valid_original().shape[1] / 4)), None)
            source_x_valid_nones = np.full((source_y_valid_nones.shape[0], source_y_valid_nones.shape[1] - 2), None)
            target_x_valid_nones = np.full((source_y_valid_nones.shape[0], source_y_valid_nones.shape[1] - 2), None)


        else:
            source_y_valid_nones = np.full((data_handler.get_source_y_valid_original().shape[0],
                                            data_handler.get_source_y_valid_original().shape[1] + int(
                                                data_handler.get_source_y_valid_original().shape[1] / 4) - 1), None)
            source_x_valid_nones = np.full(source_y_valid_nones.shape, None)
            target_x_valid_nones = np.full(source_y_valid_nones.shape, None)

        source_pred_valid_nones = np.full(source_y_valid_nones.shape, None)
        target_pred_valid_nones = np.full(source_y_valid_nones.shape, None)


        for i in range(0, source_x_train.shape[1], 3):
            a = i + (int(i / 3) * 2)
            # print(data_handler.get_source_x_train_original().shape)
            source_x_train_nones[:, a:a + 3] = data_handler.get_source_x_train_original()[:, i:i + 3]
            target_x_train_nones[:, a:a + 3] = data_handler.get_target_x_train_original()[:, i:i + 3]
            source_x_valid_nones[:, a:a + 3] = data_handler.get_source_x_valid_original()[:, i:i + 3]
            target_x_valid_nones[:, a:a + 3] = data_handler.get_target_x_valid_original()[:, i:i + 3]

        for i in range(0, source_y_train.shape[1], 4):
            a = i + int(i / 4)
            source_y_train_nones[:, a:a + 4] = data_handler.get_source_y_train_original()[:, i:i + 4]
            source_y_valid_nones[:, a:a + 4] = data_handler.get_source_y_valid_original()[:, i:i + 4]
            # print(predictions['source_train'].shape)
            source_pred_train_nones[:, a:a + 4] = predictions['source_train'][:, i:i + 4]
            target_pred_train_nones[:, a:a + 4] = predictions['target_train'][:, i:i + 4]
            source_pred_valid_nones[:, a:a + 4] = predictions['source_valid'][:, i:i + 4]
            target_pred_valid_nones[:, a:a + 4] = predictions['target_valid'][:, i:i + 4]

        print("ho")
        print(source_pred_train_nones.shape)
        print(predictions['source_train'].shape)
        print(source_pred_train_nones[:, -2:].shape)
        print(predictions['source_train'][:, -2:].shape)
        # source_pred_train_nones[:, -2:] = predictions['source_train'][:, -2:]
        # target_pred_train_nones[:, -2:] = predictions['target_train'][:, -2:]
        # source_pred_valid_nones[:, -2:] = predictions['source_valid'][:, -2:]
        # target_pred_valid_nones[:, -2:] = predictions['target_valid'][:, -2:]

        if NUMBER_OF_PREDICTION_PLOTS > -1:
            for ind in rnd_train_indices[:NUMBER_OF_PREDICTION_PLOTS]:
                prediction_plot_nones(source_x=source_x_train_nones,
                                      source_y=source_y_train_nones,
                                      target_x=target_x_train_nones,
                                      ind=ind,
                                      predict_source=source_pred_train_nones,
                                      predict_target=target_pred_train_nones,
                                      title='Training',
                                      labels=labels_train)
                plt.savefig('plots/STEATOSIS_pred_720LE_L1O/' + PLOT_PREFIX + 'train' + str(ind))

            for ind in rnd_valid_indices[:NUMBER_OF_PREDICTION_PLOTS]:
                prediction_plot_nones(source_x=source_x_valid_nones,
                                      source_y=source_y_valid_nones,
                                      target_x=target_x_valid_nones,
                                      ind=ind,
                                      predict_source=source_pred_valid_nones,
                                      predict_target=target_pred_valid_nones,
                                      title='Validation',
                                      labels=labels_valid)
                plt.savefig('plots/STEATOSIS_pred_720LE_L1O/' + PLOT_PREFIX + 'valid' + str(ind) + str(i_one))

            plt.show()
        else:
            for ind in range(len(source_x_train_nones)):
                prediction_plot_nones(source_x=source_x_train_nones,
                                      source_y=source_y_train_nones,
                                      target_x=target_x_train_nones,
                                      ind=ind,
                                      predict_source=source_pred_train_nones,
                                      predict_target=target_pred_train_nones,
                                      title='Training',
                                      labels=labels_train)
                plt.savefig('plots/STEATOSIS_pred_720LE_L1O/' + PLOT_PREFIX + 'train' + str(ind))
                plt.close()

            for ind in range(len(source_x_valid_nones)):
                prediction_plot_nones(source_x=source_x_valid_nones,
                                      source_y=source_y_valid_nones,
                                      target_x=target_x_valid_nones,
                                      ind=ind,
                                      predict_source=source_pred_valid_nones,
                                      predict_target=target_pred_valid_nones,
                                      title='Validation',
                                      labels=labels_valid)
                plt.savefig('plots/STEATOSIS_pred_720LE_L1O/' + PLOT_PREFIX + 'valid' + str(ind)+ str(i_one))
                plt.close()
