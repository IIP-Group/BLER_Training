# -----------------------------------------------------
# -- Simulator for AWGN experiment with SNR de-weighting training
# -- October 2022 (c) Reinhard Wiesmayr (wiesmayr@iis.ee.ethz.ch)
# -----------------------------------------------------
# If you use this simulator, then you must cite our paper:
# - R. Wiesmayr, G. Marti, C. Dick, H. Song, and C. Studer "Bit Error and Block Error Rate Training for ML-Assisted
# Communication," arXiv:2210.14103, 2022, available at https://arxiv.org/abs/2210.14103

import tensorflow as tf
import sionna
from tensorflow.keras import Model

from source.dampenedLdpc5gDecoder import dampedLDPC5GDecoder
import numpy as np

import matplotlib.pyplot as plt
# Load the required sionna components
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.utils import BinarySource, ebnodb2no, sim_ber, expand_to_rank
from sionna.channel import AWGN

from source.simulationFunctions import save_weights, load_weights, save_data, train_model, \
    train_model_deweighting_SNR

# Simulation setup
case = "awgn-chan_snr_deweighting_bg1-2k5training"

num_ldpc_iter = 5   # only 5 LDPC iterations
GPU_NUM = 0
DEBUG = False
XLA_ENA = False
OPTIMIZED_LDPC_INTERLEAVER = True

# Set seeds (TF and NP)
tf.random.set_seed(1)
np.random.seed(1)

# Select GPU 0 to run TF/Sionna
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = GPU_NUM  # Number of the GPU to be used
    try:
        # tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

# simulation parameters
batch_size = int(4e4)  # number of symbols to be analyzed
num_iter = 250  # number of Monte Carlo Iterations (total number of Monte Carlo runs is num_iter*batch_size)

num_pretraining_iterations = 2500
num_snr_training_epochs = 10
num_iter_per_epoch = 250

training_batch_size = int(200)
ebno_db_min = 1  # min and max EbNo value in dB for training and SNR de-weighting
ebno_db_max = 7
# path where to save the learnt parameters
stepsize = 0.25

if DEBUG:
    batch_size = int(1e1)
    num_iter = 2
    num_pretraining_iterations = 10
    num_BLER_training_iterations = 10
    num_iter_per_epoch = 5
    stepsize = 5
    training_batch_size = int(2e0)
    tf.config.run_functions_eagerly(True)
    mi_stepsize = 0.25
    rb_used = 2
else:
    tf.config.run_functions_eagerly(False)
    sionna.config.xla_compat = XLA_ENA

snr_range=np.arange(0, 8+stepsize, stepsize)

low_complexity = True

if low_complexity:
    demapping_method = "maxlog"
    ldpc_cn_update_func = "minsum"
else:
    demapping_method = "app"
    ldpc_cn_update_func = "boxplus"

num_bits_per_symbol = 2     # QPSK

# LDPC code parameters - high puncturing, high rate, short code, base graph 1
r = 5.0/6.0  # rate 5/6
k = 300  # number of information bits per codeword (292 is minimum for BG1)
n = int(k/r)  # code length (most probably selects the largest 5G generator matrix

model_weights_path = '../data/weights/' + case + "_" + str(ebno_db_min) + "_" + str(ebno_db_max) + "_"


# Constellation 4-QAM
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

# father class including utility functions
class BaseModel(Model):
    def __init__(self, num_bp_iter=5, loss_fun="BCE", training=False):
        super().__init__()
        num_bp_iter = int(num_bp_iter)
        self._lossFun = loss_fun
        self._training = training

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=num_bits_per_symbol)
        self._mapper = Mapper(constellation=constellation)

        ######################################
        ## Channel
        self._channel = AWGN()

    def computeLoss(self, b, c, b_hat):
        if self._training:
            cost = 0
            if 'BCE' in self._lossFun:
                bce = tf.nn.sigmoid_cross_entropy_with_logits(b, b_hat)
                if 'LogSumExp' in self._lossFun:
                    if 'normalized' in self._lossFun:
                        x_max = tf.reduce_max(bce, axis=-1, keepdims=True)
                        x_max_ = tf.squeeze(x_max, axis=-1)
                        cost = x_max_ + tf.math.log(
                            tf.math.reduce_sum(tf.exp(bce - x_max), axis=-1) - tf.exp(-x_max_) * tf.cast(
                                tf.shape(bce)[-1] - 1, tf.float32))
                    else:
                        cost = tf.math.reduce_logsumexp(bce, axis=-1)
                elif 'pNorm_2' in self._lossFun:
                    if 'noRoot' in self._lossFun:
                        cost = tf.pow(bce, 2)
                    else:
                        bce = tf.concat([bce, expand_to_rank(tf.ones(tf.shape(bce)[:-1])*(1e-4), tf.rank(bce))], axis=-1)
                        cost = tf.norm(bce, ord=2, axis=-1)
                elif 'pNorm_4' in self._lossFun:
                    if 'noRoot' in self._lossFun:
                        cost = tf.pow(bce, 4)
                    else:
                        bce = tf.concat([bce, expand_to_rank(tf.ones(tf.shape(bce)[:-1])*(1e-4), tf.rank(bce))], axis=-1)
                        cost = tf.norm(bce, ord=4, axis=-1)
                elif 'max' in self._lossFun:
                    cost = tf.math.reduce_max(bce, axis=-1)
                elif 'softMax' in self._lossFun:
                    # alpha = 0 corresponds to arithmetic mean (standard BCE)
                    alpha = 0.0
                    if '0_1' in self._lossFun:
                        alpha = 0.1
                    if '0_5' in self._lossFun:
                        alpha = 0.5
                    elif '1' in self._lossFun:
                        alpha = 1.0
                    elif '2' in self._lossFun:
                        alpha = 2.0
                    x_max = tf.reduce_max(bce, axis=-1, keepdims=True)
                    _exp_alpha_bce = tf.exp(alpha * (bce - x_max))
                    cost = tf.reduce_sum(bce * _exp_alpha_bce, axis=-1) / tf.reduce_sum(_exp_alpha_bce, axis=-1)
                else:
                    cost = bce
            elif 'SumLogProduct' in self._lossFun:
                cost = - tf.reduce_sum(tf.math.log(0.5 - 0.5 * tf.tanh(-b_hat * (b - 0.5))), axis=-1)
            elif 'Product' in self._lossFun:
                # cost = 1 - tf.reduce_prod(1/(1+tf.exp(b_hat*(2*c-1))), axis=-1)
                cost = 1 - tf.reduce_prod(0.5 - 0.5 * tf.tanh(-b_hat * (b - 0.5)), axis=-1)
                if 'Log' in self._lossFun:
                    cost = tf.math.log(tf.reduce_mean(cost))
            elif 'MSE' in self._lossFun:
                p = 0.5*(1-tf.tanh(-b_hat/2.0))
                cost = tf.reduce_mean(tf.pow(b-p, 2.0), axis=-1)
                # cost = tf.keras.losses.MSE(c, p)
            else:
                raise NotImplementedError('Not implemented:' + self._lossFun)

            if 'deweighting_SNR' in self._lossFun:
                cost = tf.reduce_mean(cost, axis=range(1, tf.rank(cost)))
            else:
                cost = tf.reduce_mean(cost)
            return cost
        else:
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation

    @property
    def lossFun(self):
        return self._lossFun


class AwgnModel(BaseModel):
    def __init__(self, num_bp_iter=num_ldpc_iter, training=False, loss_fun="BCE"):
        super().__init__(num_bp_iter=num_bp_iter, loss_fun=loss_fun, training=training)
        ######################################
        self._demapper = Demapper(demapping_method=demapping_method, constellation=constellation)
        self._LDPCDec0 = dampedLDPC5GDecoder(self._encoder, trainDamping=True, trainable=True,      # damped LDPC decoder is trainable
                                       cn_type=ldpc_cn_update_func, return_infobits=True,
                                       num_iter=int(num_bp_iter),
                                       hard_out=not training)

    @tf.function(jit_compile=XLA_ENA)
    def call(self, batch_size, ebno_db):

        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        # Outer coding is only performed if not training
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        b = self._binary_source([batch_size, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no = expand_to_rank(no, tf.rank(x))
        y = self._channel((x, no))

        ######################################
        ## "unbiased matched filter"

        llr_ch = self._demapper([y, no])
        b_hat = self._LDPCDec0(llr_ch)

        return self.computeLoss(b, c, b_hat)  # Ground truth and reconstructed information bits returned for BER/BLER computation

#####################################################################################################################
# Train Models

# Model for BCE pre training
training_model_BCE = AwgnModel(training=True)

loss_pretraining = train_model(training_model_BCE, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                               training_batch_size, return_loss=True)

loss_curves = {
    "loss_pretraining": loss_pretraining
}

# BLER loss functions under test
loss_fun_list = ["BCE", "MSE", "BCE_pNorm_2", 'BCE_LogSumExp_normalized', 'BCE_max', 'BCE_softMax_0_5', 'Product']
for loss_fun in loss_fun_list:
    if loss_fun != 'MSE':
        training_model_BLER = AwgnModel(training=True, loss_fun=loss_fun + "deweighting_SNR")
        training_model_BLER.set_weights(training_model_BCE.get_weights())
        loss_curve_bce = train_model_deweighting_SNR(training_model_BLER, ebno_db_min, ebno_db_max,
                                                     training_batch_size=training_batch_size,
                                                     num_training_epochs=num_snr_training_epochs,
                                                     num_iter_per_epoch=num_iter_per_epoch, return_loss=True)
        loss_curves[loss_fun] = loss_curve_bce.flatten()
    else:  # MSE loss requires pre-training with MSE
        training_model_mse = AwgnModel(training=True, loss_fun=loss_fun)
        loss_curve_bce = train_model(training_model_mse, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                                     training_batch_size, return_loss=True)
        loss_curves[loss_fun + "_pre"] = loss_curve_bce
        training_model_BLER = AwgnModel(training=True, loss_fun=loss_fun + "deweighting_SNR")
        training_model_BLER.set_weights(training_model_mse.get_weights())
        loss_curve_bce = train_model_deweighting_SNR(training_model_BLER, ebno_db_min, ebno_db_max,
                                                     training_batch_size=training_batch_size,
                                                     num_training_epochs=num_snr_training_epochs,
                                                     num_iter_per_epoch=num_iter_per_epoch, return_loss=True)
        loss_curves[loss_fun] = loss_curve_bce.flatten()
    save_weights(training_model_BLER, model_weights_path + loss_fun)

#####################################################################################################################
## Define Benchmark Models
#####################################################################################################################

test_model = AwgnModel(training=False)

#####################################################################################################################
## Benchmark Models
#####################################################################################################################

BLER = {'snr_range': snr_range}
BER = {'snr_range': snr_range}
#
title = case + str(num_ldpc_iter) + ' LDPC Iter ' + "MC-Iter " + str(num_iter) + str(ebno_db_min) + "_" + str(ebno_db_max)

ber, bler = sim_ber(test_model, ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
BLER["default"] = bler.numpy()
BER["default"] = ber.numpy()

for loss_fun in loss_fun_list:
    test_model = AwgnModel(training=False)
    test_model = load_weights(test_model, model_weights_path + loss_fun)
    ber, bler = sim_ber(test_model, ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
    BLER[loss_fun] = bler.numpy()
    BER[loss_fun] = ber.numpy()

save_data(title + "_BLER", BLER, path="../results/")
save_data(title + "_BER", BER, path="../results/")

## Plot results

keys= list(BLER.keys())
keys.remove("snr_range")
plt.figure(figsize=(10, 6))
for i in range(len(keys)):
    loss_fun = keys[i]
    plt.semilogy(snr_range, BLER[loss_fun], 'o-', c=f'C'+str(i), label=loss_fun)
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(keys)):
    loss_fun = keys[i]
    plt.semilogy(snr_range, BER[loss_fun], 'o-', c=f'C'+str(i), label=loss_fun)
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-6, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BER")
plt.show()

### Uncomment to plot the loss during training
# keys.remove("default")
# plt.figure(figsize=(10, 6))
# # plt.plot(np.arange(len(loss_pretraining)), loss_pretraining, 'o-', c=f'C'+str(i), label="Pretraining")
# for i in range(len(keys)):
#     loss_fun = keys[i]
#     plt.plot(np.arange(len(loss_curves[loss_fun].flatten())), loss_curves[loss_fun].flatten(), 'o-', c=f'C'+str(i), label=loss_fun)
# plt.xlabel(r"Iteration index")
# plt.ylabel("Loss")
# plt.grid(which="both")
# plt.legend()
# plt.tight_layout()
# plt.title(title+" Loss Curves")
# plt.show()
