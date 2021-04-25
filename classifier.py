import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mne
from fooof import FOOOF
import os
from glob import glob

from mne.time_frequency import psd_welch, psd_multitaper
from mne import Epochs, concatenate_epochs

from npa.utils import blink_removal
from npa import NPA
from keras.models import Model
from keras.layers import LSTM, Flatten, Dense, Input

from keras.constraints import max_norm
from keras.optimizers import Adam, SGD
from keras.utils import Sequence, to_categorical

import keras.backend as K

from sklearn.model_selection import GroupKFold

import h5py, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap


data_dir = '/home/bchak/github/course/pybids/examples/ds000117'

subjects = ['sub-{:02}'.format(i) for i in range(1, 2)]
sessions = dict()

for participant_idx, participant in enumerate(subjects):
    sessions[participant] = []
    for run_idx, filename in enumerate(glob(data_dir + participant + 'ses-meg/meg/' + participant + '_ses-meg_task-facerecognition_run-*_meg.fif')):
        sessions[participant].append(filename[-10:-9])

preproc_types = ['NPA', 'Bandpass', 'Highpass', 'Raw']

print('Subjects:', subjects)

n_channels = 60
n_timepoints = 456
eeg_ch_names = ['EEG{:03}'.format(i) for i in range(1, 61)]

plot_colours = ['blue', 'red', 'green', 'darkorange']


def remove_duplicate_events(events):
    new_events = []
    new_events.append(events[0])
    for event in events[1:]:
        if event[0] != new_events[-1][0]:
            new_events.append(event)

    return new_events

def convert_epochs_float32(epochs):
    epoch_data = epochs.get_data()

    min, max = np.min(epoch_data), np.max(epoch_data)
    epoch_data = (epoch_data - min) / (max - min)

    epoch_data_float32 = np.float32(np.copy(epoch_data))

    new_epochs = mne.EpochsArray(epoch_data_float32, epochs.info, verbose=0)

    return new_epochs

def save_epochs_as(eeg, preproc_type, events, participant, session_name):
    os.makedirs(data_dir + '/epochs/' + preproc_type, exist_ok=True)

    familiar_events = mne.pick_events(events, include=[5, 6, 7])
    familiar_events = remove_duplicate_events(familiar_events)

    familiar_epochs = Epochs(eeg, familiar_events, tmin=-0.3, tmax=1, proj=True, detrend=0, preload=False, verbose=0).drop_bad()
    familiar_epochs = convert_epochs_float32(familiar_epochs)
    familiar_epochs.save(data_dir + '/epochs/' + preproc_type + '/familiar_' + participant + '_' + session_name + '-epo.fif', verbose=0)

    unfamiliar_events = mne.pick_events(events, include=[13, 14, 15])
    unfamiliar_events = remove_duplicate_events(unfamiliar_events)

    unfamiliar_epochs = Epochs(eeg, unfamiliar_events, tmin=-0.3, tmax=1, proj=True, detrend=0, preload=False, verbose=0).drop_bad()
    unfamiliar_epochs = convert_epochs_float32(unfamiliar_epochs)
    unfamiliar_epochs.save(data_dir + '/epochs/' + preproc_type + '/unfamiliar_' + participant + '_' + session_name + '-epo.fif', verbose=0)

    noise_events = mne.pick_events(events, include=[17, 18, 19])
    noise_events = remove_duplicate_events(noise_events)

    noise_epochs = Epochs(eeg, noise_events, tmin=-0.3, tmax=1, proj=True, detrend=0, preload=False, verbose=0).drop_bad()
    noise_epochs = convert_epochs_float32(noise_epochs)
    noise_epochs.save(data_dir + '/epochs/' + preproc_type + '/noise_' + participant + '_' + session_name + '-epo.fif', verbose=0)

    return familiar_epochs, unfamiliar_epochs, noise_epochs

def load_montage(montage_file):

    channels = []
    pos = []

    with open(montage_file, 'r') as f:
        for line in f:
            if len(line) > 3:
                tokens = line.split(' ')

                if 'EEG' in tokens[0]:
                    channels.append(tokens[0])

                    x = float(tokens[1])
                    y = float(tokens[2])
                    z = float(tokens[3])

                    pos.append([x, y, z])

    montage = mne.channels.Montage(np.asarray(pos), channels,'standard_1020', list(range(len(channels))))

    return montage

def process_subject(data_file, subject, session):

    # montage_file = data_dir + '/' + subject + '/ses-meg/meg/' + subject + '_ses-meg_headshape.pos'
    # montage = load_montage(montage_file)

    meg = mne.io.read_raw_fif(data_file, preload=True, verbose=0)
    # meg.set_montage(montage)

    events = mne.find_events(meg, shortest_event=0, stim_channel=['STI101'], verbose=0)
    # 17,18,19 are noise
    # 5,6,7 are familiar faces
    # 13,14,15 are unfamiliar faces

    sampling_frequency = 350

    eeg = meg.pick_types(meg=False, eeg=True, stim=False, eog=True, exclude='bads', selection=None, verbose=0)

    # eeg.plot_psd()

    eeg, events = eeg.resample(sampling_frequency, n_jobs=-1, events=events, verbose=0)
    eeg.info['sfreq'] = 350

    eeg_ch_names = ['EEG{:03}'.format(i) for i in range(1, 61)]
    eeg_ch_ints = [eeg.ch_names.index(ch_name) for ch_name in eeg_ch_names]

    eog_ch_names = ['EEG061', 'EEG062']

    eog_ch_ints = [eeg.ch_names.index(eog_ch_name) for eog_ch_name in eog_ch_names]
    eeg = eeg.notch_filter(freqs=np.arange(50, sampling_frequency/2, 50), n_jobs=-1, phase='zero')
    eeg = blink_removal(eeg, eeg_ch_ints, eog_ch_ints)

    save_epochs_as(eeg, 'Raw', events, subject, session)

    bandpass_eeg = eeg.copy()
    bandpass_eeg = bandpass_eeg.filter(1, 40, n_jobs=7, phase='zero', verbose=0)

    save_epochs_as(bandpass_eeg, 'Bandpass', events, subject, session)

    highpass_eeg = eeg.copy()
    highpass_eeg = highpass_eeg.filter(1, None, n_jobs=7, phase='zero', verbose=0)

    save_epochs_as(highpass_eeg, 'Highpass', events, subject, session)

    psds, freqs = psd_multitaper(eeg, picks=eeg_ch_ints, n_jobs=-1)

    fooof = FOOOF(peak_width_limits=[3, 12], aperiodic_mode='knee')
    fooof.fit(freqs, np.mean(psds, axis=0), freq_range=[1, 45])
    fooof.plot(save_fig=True, file_name=subject+'-'+session + '.png', file_path=data_dir + '/results/')

    amp = NPA(fooof, sampling_frequency)
    amp.fit_filters(log_approx_levels=5, peak_mode='sharp', n_peak_taps=192)

    peak_fig = amp.plot_peak_filters()
    peak_fig.savefig(data_dir + '/results/' + subject + '-' + session + 'peaks.png')

    log_fig = amp.plot_log_filters()
    log_fig.savefig(data_dir + '/results/' + subject + '-' + session + 'log.png')

    # amplified_time_series = amp.amplify(eeg.get_data(picks=eeg_ch_ints))
    # amplified_eeg = eeg.copy()

    eeg = eeg.apply_function(amp.amplify, picks=eeg_ch_ints, n_jobs=-1)

    #create new MNE Raw object with amplified EEG signal
    # eeg_ch_names = [eeg.ch_names[i] for i in eeg_ch_ints]
    # amp_info = mne.create_info(eeg_ch_names, sampling_frequency, ch_types='eeg')
    # amplified_eeg = mne.io.RawArray(np.float64(amplified_time_series), amp_info)
    # amplified_eeg.set_montage(montage)

    save_epochs_as(eeg, 'NPA', events, subject, session)

    # calculate and plot PSD
    # amp_psds, amp_freqs = psd_welch(amplified_eeg, 0, 45, n_jobs=-1)
    #
    #
    # plt.plot(freqs, np.log10(np.mean(psds, axis=0)), color='blue', label='original')
    # plt.plot(amp_freqs, np.log10(np.mean(amp_psds, axis=0)), color='black', label='amplified')
    #
    # plt.xlim([0, 45])
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')
    # plt.legend()

class EEGEpochSequence(Sequence):

    def __init__(self, f, indices, batch_size):
        self.eeg = f['eeg']
        self.labels = to_categorical(f['label'])
        self.batch_size = batch_size

        self.indices = indices

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        return_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()
        # print('mean eeg', np.mean(self.eeg[return_indices[0]]), self.labels[return_indices[0]])

        return np.swapaxes(self.eeg[return_indices, ...], 1, 2), self.labels[return_indices]


def merge_all_epochs(preproc_type):
    n_familiar_epochs = 0
    n_unfamiliar_epochs = 0
    n_noise_epochs = 0

    for participant_idx, participant in enumerate(subjects):

        for filename in glob(data_dir + '/epochs/' + preproc_type + '/familiar_' + participant + '*' + '-epo.fif'):
            familiar_epochs = mne.read_epochs(filename, proj=False, preload=False, verbose=0)
            n_familiar_epochs += len(familiar_epochs)

        for filename in glob(data_dir + '/epochs/' + preproc_type + '/unfamiliar_' + participant + '*' + '-epo.fif'):
            unfamiliar_epochs = mne.read_epochs(filename, proj=False, preload=False, verbose=0)
            n_unfamiliar_epochs += len(unfamiliar_epochs)

        for filename in glob(data_dir + '/epochs/' + preproc_type + '/noise_' + participant + '*' + '-epo.fif'):
            noise_epochs = mne.read_epochs(filename, proj=False, preload=False, verbose=0)
            n_noise_epochs += len(noise_epochs)

    n_total_epochs = n_familiar_epochs + n_unfamiliar_epochs + n_noise_epochs

    print('Familiar epochs:', n_familiar_epochs)
    print('Unfamiliar epochs:', n_unfamiliar_epochs)
    print('Noise epochs:', n_noise_epochs)

    print('Total epochs in dataset:', n_total_epochs)

    f = h5py.File(data_dir + '/epochs/' + preproc_type + '.hdf5', 'w')
    f.create_dataset('eeg', (n_total_epochs, n_channels, n_timepoints), dtype='float32')
    f.create_dataset('label', (n_total_epochs,), dtype='uint8')
    f.create_dataset('participant', (n_total_epochs,), dtype='uint8')


    idx = 0
    for participant_idx, participant in enumerate(subjects):
        for filename in glob(data_dir + '/epochs/' + preproc_type + '/familiar_' + participant + '*' + '-epo.fif'):
            familiar_epochs = mne.read_epochs(filename, proj=False, preload=True, verbose=0).get_data()

            print(familiar_epochs.shape)

            familiar_epochs = np.asarray(familiar_epochs, dtype='float32')

            participant_num = participant_idx + 1

            f['eeg'][idx:idx + familiar_epochs.shape[0], 0:n_channels, :] = familiar_epochs[:, 0:n_channels, :]
            f['participant'][idx:idx + familiar_epochs.shape[0]] = participant_num
            f['label'][idx:idx + familiar_epochs.shape[0]] = 0

            idx += familiar_epochs.shape[0]

        for filename in glob(data_dir + '/epochs/' + preproc_type + '/unfamiliar_' + participant + '*' + '-epo.fif'):
            unfamiliar_epochs = mne.read_epochs(filename, proj=False, preload=True, verbose=0).get_data()

            unfamiliar_epochs = np.asarray(unfamiliar_epochs, dtype='float32')

            f['eeg'][idx:idx + unfamiliar_epochs.shape[0], 0:n_channels, :] = unfamiliar_epochs[:, 0:n_channels, :]
            f['participant'][idx:idx + unfamiliar_epochs.shape[0]] = participant_num
            f['label'][idx:idx + unfamiliar_epochs.shape[0]] = 1

            idx += unfamiliar_epochs.shape[0]

        for filename in glob(data_dir + '/epochs/' + preproc_type + '/noise_' + participant + '*' + '-epo.fif'):
            noise_epochs = mne.read_epochs(filename, proj=False, preload=True, verbose=0).get_data()

            noise_epochs = np.asarray(noise_epochs, dtype='float32')

            f['eeg'][idx:idx + noise_epochs.shape[0], 0:n_channels, :] = noise_epochs[:, 0:n_channels, :]
            f['participant'][idx:idx + noise_epochs.shape[0]] = participant_num
            f['label'][idx:idx + noise_epochs.shape[0]] = 2

            idx += noise_epochs.shape[0]

    f.close()


def plot_grouped_evoked():

    for participant_idx, participant in enumerate(subjects):
        evoked_fig, evoked_ax = plt.subplots(nrows=1, ncols=len(preproc_types), sharex=True, sharey=True, squeeze=False, figsize=(24, 6))

        for preproc_idx, preproc_type in enumerate(preproc_types):
            familiar_evoked_data = np.zeros((n_channels, n_timepoints))
            unfamiliar_evoked_data = np.zeros((n_channels, n_timepoints))
            # noise_evoked_data = np.zeros((n_channels, n_timepoints))

            runs = 0
            for filename in glob(data_dir + '/epochs/' + preproc_type + '/familiar_' + participant + '*' + '-epo.fif'):
                familiar_epochs = mne.read_epochs(filename, proj=False, preload=True, verbose=False)
                familiar_epochs = familiar_epochs.pick_channels(eeg_ch_names)
                familiar_evoked = familiar_epochs.average()

                familiar_evoked_data += familiar_evoked.data
                runs += 1

            familiar_evoked_data = familiar_evoked_data / runs
            familiar_evoked.data = familiar_evoked_data
            familiar_evoked = familiar_evoked.detrend()
            familiar_evoked.times = familiar_evoked.times - 0.3

            runs = 0
            for filename in glob(data_dir + '/epochs/' + preproc_type + '/unfamiliar_' + participant + '*' + '-epo.fif'):
                unfamiliar_epochs = mne.read_epochs(filename, proj=False, preload=True, verbose=False)
                unfamiliar_epochs = unfamiliar_epochs.pick_channels(eeg_ch_names)
                unfamiliar_evoked = unfamiliar_epochs.average()

                unfamiliar_evoked_data += unfamiliar_evoked.data
                runs += 1

            unfamiliar_evoked_data = familiar_evoked_data / runs
            unfamiliar_evoked.data = unfamiliar_evoked_data
            unfamiliar_evoked = unfamiliar_evoked.detrend()
            unfamiliar_evoked.times = unfamiliar_evoked.times - 0.3

            # noise_evoked.plot(spatial_colors=True, time_unit='s', gfp=False, axes=evoked_ax[participant_idx][preproc_idx], window_title=None, selectable=False, show=False)

            evoked_difference = familiar_evoked.data - unfamiliar_evoked.data

            if preproc_idx == 0:
                max_npa = np.max(evoked_difference)
            max_other = np.max(evoked_difference)

            evoked_diff = familiar_evoked.copy()
            evoked_diff.data = evoked_difference * (max_npa / max_other)

            evoked_diff.plot(spatial_colors=True, time_unit='s', gfp=False, axes=evoked_ax[0][preproc_idx], window_title=None, selectable=False, show=False)

            for tick in evoked_ax[0][preproc_idx].xaxis.get_major_ticks():
                tick.label.set_fontsize(20)

            evoked_ax[0][preproc_idx].axvline(x=0, color='k', linestyle='dashed')
            evoked_ax[0][preproc_idx].axvline(x=0.17, color='darkmagenta', linestyle='dashed')
            evoked_ax[0][preproc_idx].axvline(x=0.3, color='green', linestyle='dashed')

            evoked_ax[0][preproc_idx].set_title(preproc_type, fontsize=24)
            evoked_ax[0][preproc_idx].set_xlabel('Time (s)', fontsize=20)
            evoked_ax[0][preproc_idx].set_ylabel('Voltage ($\mu$V)', fontsize=20)

        evoked_fig.savefig(data_dir + '/results/all_evoked_' + participant + '.png', dpi=500)


def lstm_model(n_channels, n_timepoints):
    inputs = Input(shape=(n_timepoints, n_channels))
    # lstm = LSTM(256, recurrent_constraint=max_norm(2.), return_sequences=True)(inputs)

    lstm = LSTM(512, recurrent_constraint=max_norm(2.))(inputs)

    # flat = Flatten()(lstm)
    output = Dense(3, activation='softmax')(lstm)

    model = Model([inputs], output)

    optimizer = Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # optimizer = SGD(lr=0.0002)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def transf_model(n_channels, n_timepoints):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = 500
    vocab_size = 20000 
    inputs = Input(shape=(n_timepoints, n_channels))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)


def train():
    n_epochs = 30
    n_folds = len(subjects)
    n_preproc_types = len(preproc_types)

    batch_size = 1024

    test_accuracies = np.zeros((n_preproc_types, n_folds), dtype='float32')
    train_accuracies = np.zeros((n_preproc_types, n_folds, n_epochs), dtype='float32')

    losses = np.zeros((n_preproc_types, n_folds, n_epochs), dtype='float32')

    model = transf_model(n_channels, n_timepoints)
    model.summary()

    results_fig, results_ax = plt.subplots(1, n_folds, figsize=(24, 4))
    loss_fig, loss_ax = plt.subplots(1, n_folds, figsize=(24, 4))

    for preproc_idx, preproc_type in enumerate(preproc_types):
        print('Beginning analysis for', preproc_type, 'pre-processing')

        with h5py.File(data_dir + '/epochs/' + preproc_type + '.hdf5', 'r') as f:

            labels = f['label']
            participant_nums = f['participant']
            all_indices = np.asarray(range(len(labels)))

            print('labels:', set(labels), len(labels))
            print('participants:', set(participant_nums), len(participant_nums))
            print('indices:', all_indices, len(all_indices))

            gkf = GroupKFold(n_splits=n_folds)
            for fold_idx, (train_indices, test_indices) in enumerate(gkf.split(all_indices, labels, participant_nums)):
                print('Training', preproc_type, 'fold', str(fold_idx+1), '/', str(n_folds))

                model = transf_model(n_channels, n_timepoints)

                eeg_seq_train = EEGEpochSequence(f, train_indices, batch_size)
                eeg_seq_test = EEGEpochSequence(f, test_indices, batch_size)

                history = model.fit_generator(eeg_seq_train, epochs=n_epochs, steps_per_epoch=train_indices.shape[0] // batch_size, shuffle=True, use_multiprocessing=False)

                metrics = model.evaluate_generator(eeg_seq_test)
                print(model.metrics_names, metrics)

                test_accuracies[preproc_idx, fold_idx] = metrics[1]

                train_accuracies[preproc_idx, fold_idx, :] = np.copy(history.history['acc'])
                losses[preproc_idx, fold_idx, :] = np.copy(history.history['loss'])

            print('Results for preprocessing type:', preproc_type)
            print('Fold testing accuracies:', test_accuracies)

        K.clear_session()

    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])

    
    print('Results:')
    for preproc_idx, preproc_type in enumerate(preproc_types):
        print('Pre-processing type:', preproc_type, 'train accuracy:', np.mean(train_accuracies[preproc_idx, :, -1]), 'test accuracy:', np.mean(test_accuracies[preproc_idx, :]))

        for fold_idx in range(n_folds):

            results_ax[fold_idx].plot(train_accuracies[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx], label=preproc_type)
            loss_ax[fold_idx].plot(losses[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx], label=preproc_type)

            results_ax[fold_idx].legend(loc='center right', shadow=True, fancybox=True)

            results_ax[fold_idx].set_xlabel('Epoch', fontsize=16)
            results_ax[fold_idx].set_ylabel('Train Accuracy', fontsize=16)
            # results_ax[fold_idx].set_ylim([0.45, 1.05])

            loss_ax[fold_idx].legend(shadow=True, fancybox=True)

            loss_ax[fold_idx].set_xlabel('Epoch', fontsize=16)
            loss_ax[fold_idx].set_ylabel('Loss', fontsize=16)

    results_fig.savefig(data_dir + '/results/decoding_results.png', dpi=500, bbox_inches='tight')
    loss_fig.savefig(data_dir + '/results/loss.png', dpi=500, bbox_inches='tight')


    test_results_fig, test_results_ax = plt.subplots(1, 1, figsize=(4, 3))

    boxplots = test_results_ax.boxplot(test_accuracies.T, labels=preproc_types, patch_artist=True)
    for patch, colour in zip(boxplots['boxes'], plot_colours):
        patch.set_facecolor(colour)

    test_results_ax.set_xlabel('Pre-Processing Method', fontsize=16)
    test_results_ax.set_ylabel('Test Accuracy', fontsize=16)
    test_results_ax.grid(b=True, which='both')

    test_results_fig.savefig(data_dir + '/results/test_scores.png', dpi=500, bbox_inches='tight')





if __name__ == '__main__':
    for subject_idx, subject in enumerate(subjects):
       subject_dir = data_dir + '/' + subject + '/ses-meg/meg/'

       for run_idx, run_file in enumerate(glob(subject_dir + subject + '_ses-meg_task-facerecognition_run-*_meg.fif')):
           process_subject(run_file, subject, 'session-' + str(run_idx + 1))

    # plot_grouped_evoked()

    os.makedirs(data_dir + '/results/', exist_ok=True)

    for preproc in preproc_types:
        merge_all_epochs(preproc)

    train()

