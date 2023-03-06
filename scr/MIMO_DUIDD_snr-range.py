# -----------------------------------------------------
# -- Simulator for MU-MIMO OFDM experiment with SNR range-training
# -- October 2022 (c) Reinhard Wiesmayr (wiesmayr@iis.ee.ethz.ch)
# -----------------------------------------------------
# If you use this simulator, then you must cite our paper:
# - R. Wiesmayr, G. Marti, C. Dick, H. Song, and C. Studer "Bit Error and Block Error Rate Training for ML-Assisted
# Communication," arXiv:2210.14103, 2022, available at https://arxiv.org/abs/2210.14103

import tensorflow as tf
import sionna
from sionna.channel.tr38901 import PanelArray, UMi, UMa, RMa
from tensorflow.keras import Model
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, RemoveNulledSubcarriers, PilotPattern

from source.dampenedLdpc5gDecoder import dampedLDPC5GDecoder
import numpy as np

import matplotlib.pyplot as plt
# Load the required sionna components
from sionna.mapping import Constellation, Mapper
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.utils import BinarySource, ebnodb2no, sim_ber, expand_to_rank
from sionna.channel import RayleighBlockFading, OFDMChannel
from source.modified_single_sector_topology_generation import gen_single_sector_topology

from source.mmsePIC import sisoLowComplexityMfLmmseDetector, SisoMmsePicDetector
from source.simulationFunctions import save_weights, load_weights, save_data, train_model

#####################################################################################################################
## Simulation Parameters
#####################################################################################################################

case = "n78-DUIDD-SNR-Rng"

# set the total number of LDPC iterations to study
num_ldpc_iter = 10
perfect_csi = False
GPU_NUM = 0

# Debug => smaller batchsize, fewer training and Monte Carlo iterations
DEBUG = False
channel_model_str = "UMa"  # None for REMCOM; alternative "UMi", "UMa", "RMa"
normalizing_channels = True
XLA_ENA = False
OPTIMIZED_LDPC_INTERLEAVER = True

# LoS True only line of sight, False: none-los, none: mix of los and none-los
LoS = True
Antenna_Array = "Dual-Pol-ULA"
MOBILITY = True

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

# Set seeds (TF and NP)
tf.random.set_seed(1)
np.random.seed(1)

# simulation parameters
batch_size = int(4e3)  # number of symbols to be analyzed
num_iter = 25  # number of Monte Carlo Iterations (total number of Monte Carlo runs is num_iter*batch_size)

num_pretraining_iterations = 2500
num_BLER_training_iterations = 2500

training_batch_size = int(200)  # Batchsize per backpropagation step
ebno_db_min = -10  # min EbNo value in dB for training
ebno_db_max = 10

rb_used = 2
stepsize = 1
if DEBUG:
    batch_size = int(1e2)
    num_iter = 2
    num_pretraining_iterations = 10
    num_BLER_training_iterations = 10
    num_iter_per_epoch = 5
    stepsize = 5
    training_batch_size = int(5e1)
    tf.config.run_functions_eagerly(True)
    mi_stepsize = 0.25
    rb_used = 2
else:
    tf.config.run_functions_eagerly(False)
    sionna.config.xla_compat = XLA_ENA

# OFDM Waveform Settings
# 60kHz subcarrier spacing
subcarrier_spacing = 60e3
# Maximum resource blocks within the 100MHz n78 5G channel w/ 60MHz SC spacing
num_resource_blocks = 135
# with 30MHz sc spacing
# num_resource_blocks = 273
num_subcarriers = num_resource_blocks * 12
# we use only a subset of resource blocks (rb_used) for our simulations (here, some smaller number 2-15). As we want
# short code lengths, we select rb_used=2
sc_used = rb_used * 12  # 5G standard
guard_carriers = [10, 9]  # selected a larger number of guard bands for even shorter code lengths
dc_null = True

low_complexity = True

carrier_freq = 3.75e9   # Center frequency of 3.7GHz-3.8GHz n78 100MHz band
# bandwidth = 100e6
# effective_bandwidth = subcarrier_spacing * sc_used
# 14 OFDM TIME symbols is one 5G OFDM frame
num_ofdm_symbols = 14
pilot_ofdm_symbol_indices=[2,3,11,12]   # in the beginning and at the end of the OFDM frame to compensate for mobility
num_pilot_symbols = len(pilot_ofdm_symbol_indices)

if low_complexity:
    demapping_method = "maxlog"
    ldpc_cn_update_func = "minsum"
else:
    demapping_method = "app"
    ldpc_cn_update_func = "boxplus"

snr_range=np.arange(-10, 20+stepsize, stepsize)
num_bits_per_symbol = 2  # bits per modulated symbol, i.e., 2^2 QPSK, 2^4 = 16-QAM
_num_const_bits_ldpc = num_bits_per_symbol
if not OPTIMIZED_LDPC_INTERLEAVER:
    _num_const_bits_ldpc = None
num_streams_per_tx = 1
n_ue = 4        # 4 UEs
n_bs_ant = 16  # 16 antenna BS

data_sc_used = sc_used - np.sum(np.array(guard_carriers)) - int(dc_null)  # number of data carrying subcarriers
# Pilot pattern and OFDM resource grid will be plotted later
mask = np.zeros([   n_ue, 1, num_ofdm_symbols, data_sc_used], bool)
mask[...,pilot_ofdm_symbol_indices,:] = True
pilots = np.zeros([n_ue, 1, np.sum(mask[0,0])])
pilots[0,0, 0*data_sc_used:1*data_sc_used:2] = 1
pilots[0,0, 1+2*data_sc_used:3*data_sc_used:2] = 1
pilots[1,0,1+0*data_sc_used:1*data_sc_used:2] = 1
pilots[1,0,2*data_sc_used:3*data_sc_used:2] = 1
pilots[2,0, 1*data_sc_used:2*data_sc_used:2] = 1
pilots[2,0, 1+3*data_sc_used:4*data_sc_used:2] = 1
pilots[3,0,1+1*data_sc_used:2*data_sc_used:2] = 1
pilots[3,0,3*data_sc_used:4*data_sc_used:2] = 1
pilot_pattern = PilotPattern(mask, pilots, normalize=True)

if channel_model_str in ["UMi", "UMa", "RMa"]:
    # BS antenna configuration
    bs_array = None
    if Antenna_Array == "Dual-Pol-ULA":     # that's what we selected
        bs_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=int(n_bs_ant/2),
                              polarization='dual',
                              polarization_type='cross',
                              antenna_pattern='38.901',
                              carrier_frequency=carrier_freq)
    else:
        bs_array = PanelArray(num_rows_per_panel=int(n_bs_ant/2/8),
            num_cols_per_panel = 8,
            polarization = 'dual',
            polarization_type = 'cross',
            antenna_pattern = '38.901',
            carrier_frequency = carrier_freq)
    # UE antenna configuration
    ut_array = PanelArray(num_rows_per_panel=1,
        num_cols_per_panel = 1,
        polarization = 'single',
        polarization_type = 'V',
        antenna_pattern = 'omni',
        carrier_frequency = carrier_freq)
elif channel_model_str =="Rayleigh":
    # channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=n_bs_ant, num_tx=n_ue, num_tx_ant=1)
    # channel_model_str = " Frequency-Flat Rayleigh-Block-Fading "
    pass
else:
    raise NameError('channel_model_string not found')

# path were we store the trained model weights
model_weights_path = '../data/weights/' + case + str(perfect_csi) + channel_model_str + "_" + str(ebno_db_min) + "_" + \
                     str(ebno_db_max) + "_" + str(n_bs_ant) + "x" + str(n_ue) + case
# 50 km/h maximum UE velocity, generated uniformly random
max_ut_velocity = 50.0/3.6
if not MOBILITY:
    max_ut_velocity = 0

# LDPC code parameters
r = 0.75  # rate 3/4
n = int(
    (sc_used - np.sum(np.array(guard_carriers)) - int(dc_null)) *
    (num_ofdm_symbols - num_pilot_symbols) *
    num_bits_per_symbol)  # code length, resulting from OFDM resource configuration
k = int(n * r)  # number of information bits per codeword

# Constellation 4 QAM
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

# Define MU-MIMO System
rx_tx_association = np.zeros([1, n_ue])
rx_tx_association[0, :] = 1

# stream management stores a mapping from Rx and Tx
sm = StreamManagement(rx_tx_association, num_streams_per_tx)


rg_chan_est = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=sc_used, num_guard_carriers=guard_carriers,
                           dc_null=dc_null, subcarrier_spacing=subcarrier_spacing,
                           cyclic_prefix_length=20, num_tx=n_ue, pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
                           num_streams_per_tx=num_streams_per_tx, pilot_pattern=pilot_pattern)

# plot resource grid
rg_chan_est.show()
plt.show()
# plot pilot pattern
rg_chan_est.pilot_pattern.show()
plt.show()
#####################################################################################################################
## Define Models
#####################################################################################################################
# father class for different models, which, e.g., the 2 iterations MMSE PIC CEst model inherits from
class ChanEstBaseModel(Model):
    def __init__(self, num_bp_iter=5, perfect_csi=perfect_csi, loss_fun="BCE", training=False):
        super().__init__()
        num_bp_iter = int(num_bp_iter)
        self._lossFun = loss_fun
        self._training = training

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=_num_const_bits_ldpc)
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(rg_chan_est)

        ######################################
        ## Channel
        if channel_model_str == "UMi":
            self._channel_model = UMi(carrier_frequency=carrier_freq,
                                      o2i_model='low',
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction='uplink')
        elif channel_model_str == "UMa":
            self._channel_model = UMa(carrier_frequency=carrier_freq,
                                      o2i_model='low',
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction='uplink')
        elif channel_model_str == "RMa":
            self._channel_model = RMa(carrier_frequency=carrier_freq,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction='uplink')
        elif channel_model_str == "Rayleigh":
            self._channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=n_bs_ant, num_tx=n_ue, num_tx_ant=1)
        self._channel = OFDMChannel(channel_model=self._channel_model,
                                    resource_grid=rg_chan_est,
                                    add_awgn=True,
                                    normalize_channel=normalizing_channels, return_channel=True)

        ######################################
        ## Receiver
        self._perfect_csi = perfect_csi
        if self._perfect_csi:
            self._removeNulldedSc = RemoveNulledSubcarriers(rg_chan_est)
        else:
            self._ls_est = LSChannelEstimator(rg_chan_est, interpolation_type="lin")

    def new_topology(self, batch_size):
        """Set new topology"""
        if channel_model_str in ["UMi", "UMa", "RMa"]:
            # sensible values according to 3GPP standard, no mobility by default
            [ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state] = \
                gen_single_sector_topology(batch_size,
                                           n_ue, max_ut_velocity=max_ut_velocity,
                                           scenario=channel_model_str.lower())
            if LoS is not None:
                in_state = tf.math.logical_and(in_state, False)
            self._channel_model.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state,
                                             los=LoS)
    def computeLoss(self, b, c, b_hat):
        # tf.print(b[0, 0, 0, 0])
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

# DUIDD receiver with 2 unfolded IDD iterations, MMSE PIC, LDPC and channel estimation
class TwoIterMmsePicLdpcChanEstModel(ChanEstBaseModel):
    def __init__(self, training=False, num_bp_iter=int(num_ldpc_iter/2), perfect_csi=perfect_csi, loss_fun='BCE'):
        super().__init__(num_bp_iter=num_bp_iter, perfect_csi=perfect_csi, loss_fun=loss_fun, training=training)

        # special trainable low complexity detector, instead of initial LMMSE detection. Refer to the paper
        # R. Wiesmayr et al., "DUIDD: Deep-Unfolded Interleaved Detection and Decoding for for MIMO Wireless Systems"
        # from the 2022 ASILOMAR Conference
        self._detector0 = sisoLowComplexityMfLmmseDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                           constellation=constellation, low_complexity=low_complexity,
                                                           trainable=training)
        # SISO MMSE PIC Detector
        self._detector1 = SisoMmsePicDetector(rg_chan_est, sm, demapping_method=demapping_method,
                                                 constellation=constellation, low_complexity=low_complexity,
                                                  data_carrying_whitened_inputs=True)
        self._LDPCDec0 = dampedLDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func, return_infobits=False,
                                               num_iter=int(num_bp_iter), stateful=True, trainable=False,
                                               trainDamping=training,  hard_out=False)
        self._LDPCDec1 = dampedLDPC5GDecoder(self._encoder, cn_type=ldpc_cn_update_func,
                                               return_infobits=True,
                                               num_iter=int(num_bp_iter), stateful=True, trainable=False,
                                               trainDamping=training,
                                               hard_out=not (self._training))
        self._alpha1 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha1")
        self._alpha2 = tf.Variable(1, trainable=training, dtype=tf.float32, name="alpha2")

        self._beta1 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta1")
        self._beta2 = tf.Variable(0, trainable=training, dtype=tf.float32, name="beta2")

        self._gamma1 = tf.Variable(0.0, trainable=training, dtype=tf.float32, name="gamma1")

        self._eta1 = tf.Variable(1.0, trainable=training, dtype=tf.float32, name="eta1")

    @property
    def eta1(self):
        return self._eta1

    @property
    def gamma1(self):
        return self._gamma1

    @property
    def alpha1(self):
        return self._alpha1

    @property
    def beta1(self):
        return self._beta1

    @property
    def alpha2(self):
        return self._alpha2

    @property
    def beta2(self):
        return self._beta2

    @tf.function(jit_compile=XLA_ENA)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, r)
        b = self._binary_source([batch_size, n_ue, num_streams_per_tx, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        if self._perfect_csi:
            h_hat = self._removeNulldedSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat), dtype=tf.float32)  # No channel estimation error when perfect CSI
            # knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est([y, no])

        chan_est_var = self.eta1 * chan_est_var

        [llr_ch, _, G, y_MF, _, h_dt_desired_whitened] = self._detector0(
            [y, h_hat, chan_est_var, no, None, None, None, None])  # A_inv and G none => will calculate itself then

        [llr_dec, msg_vn] = self._LDPCDec0([llr_ch, None])
        llr_a_det = self.alpha1 * llr_dec - self.beta1 * llr_ch

        [llr_ch, _, _, _] = self._detector1([y_MF, h_dt_desired_whitened, chan_est_var, no, llr_a_det, G])
        llr_a_dec = self.alpha2 * llr_ch - self.beta2 * llr_a_det

        [b_hat, _] = self._LDPCDec1([llr_a_dec, self.gamma1*msg_vn])

        return self.computeLoss(b, c, b_hat)


#####################################################################################################################
# Train Models

# Model for BCE pre training
training_model_BCE = TwoIterMmsePicLdpcChanEstModel(training=True)
loss_pretraining = train_model(training_model_BCE, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                               training_batch_size, return_loss=True)

loss_curves = {
    "loss_pretraining": loss_pretraining
}

# BLER loss functions under test
loss_fun_list = ["BCE", "MSE", "BCE_pNorm_2", 'BCE_LogSumExp_normalized', 'BCE_max', 'BCE_softMax_0_5', 'Product']
for loss_fun in loss_fun_list:
    training_model_BLER = TwoIterMmsePicLdpcChanEstModel(training=True, loss_fun=loss_fun)
    if loss_fun != 'MSE':
        training_model_BLER.set_weights(training_model_BCE.get_weights())
        loss_curve_bce = train_model(training_model_BLER, ebno_db_min, ebno_db_max, num_BLER_training_iterations,
                                     training_batch_size, return_loss=True)
        loss_curves[loss_fun] = loss_curve_bce
        save_weights(training_model_BLER, model_weights_path + loss_fun)
    else:
        loss_curve_bce = train_model(training_model_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations,
                                     training_batch_size, return_loss=True)
        loss_curves[loss_fun + "_pre"] = loss_curve_bce
        loss_curve_bce = train_model(training_model_BLER, ebno_db_min, ebno_db_max, num_BLER_training_iterations,
                                     training_batch_size, return_loss=True)
        loss_curves[loss_fun] = loss_curve_bce
        save_weights(training_model_BLER, model_weights_path + loss_fun)

#####################################################################################################################
## Define Benchmark Models
#####################################################################################################################

test_model = TwoIterMmsePicLdpcChanEstModel(training=False)

#####################################################################################################################
## Benchmark Models
#####################################################################################################################

BLER = {'snr_range': snr_range}
BER = {'snr_range': snr_range}

title = case + 'Perfect-CSI=' + str(perfect_csi) + " " + str(n_bs_ant) + 'x' + str(n_ue) + channel_model_str + ' & ' + \
        str(num_ldpc_iter) + ' LDPC Iter ' + Antenna_Array + "MC-Iter " + str(num_iter) + "_" + str(ebno_db_min) + "_" \
        + str(ebno_db_max)

ber, bler = sim_ber(test_model, ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
BLER["default"] = bler.numpy()
BER["default"] = ber.numpy()

for loss_fun in loss_fun_list:
    test_model = TwoIterMmsePicLdpcChanEstModel(training=False)
    test_model = load_weights(test_model, model_weights_path + loss_fun)
    ber, bler = sim_ber(test_model, ebno_dbs=snr_range, batch_size=batch_size,
                        num_target_block_errors=None, max_mc_iter=num_iter, early_stop=False)
    BLER[loss_fun] = bler.numpy()
    BER[loss_fun] = ber.numpy()


save_data(title + "_BLER", BLER, path="../results/")
save_data(title + "_BER", BER, path="../results/")

## Ploting results

BLER["snr_range"] = snr_range
plt.figure(figsize=(10, 6))
keys = list(BLER.keys())
keys.remove('snr_range')
for i in range(len(keys)):
    loss_fun = keys[i]
    plt.semilogy(BLER["snr_range"], BLER[loss_fun], 'o-', c=f'C'+str(i), label=loss_fun)
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-3, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(keys)):
    loss_fun = keys[i]
    plt.semilogy(BER["snr_range"], BER[loss_fun], 'o-', c=f'C'+str(i), label=loss_fun)
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
plt.title(title+" BLER")
plt.show()

### Uncomment to plot the loss during training
# keys.pop(0)
# keys.append("MSE_pre")
# plt.figure(figsize=(10, 6))
# # plt.plot(np.arange(len(loss_pretraining)), loss_pretraining, 'o-', c=f'C'+str(i), label="Pretraining")
# for i in range(len(keys)):
#     loss_fun = keys[i]
#     plt.plot(np.arange(len(loss_curves[loss_fun])), loss_curves[loss_fun], 'o-', c=f'C'+str(i), label=loss_fun)
# plt.xlabel(r"Iteration index")
# plt.ylabel("Loss")
# plt.grid(which="both")
# plt.legend()
# plt.title(title+" Loss Curves")
# plt.tight_layout()
# plt.show()
