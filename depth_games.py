import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import scipy.signal as sp_sig
import antropy as ant
from depth_utils import *
from sklearn.preprocessing import robust_scale


# without delta features
def calc_features(epochs, subj):
    # Bandpass filter
    freq_broad = (0.1, 500)
    # FFT & bandpower parameters
    sr = 1000
    bands = [
        (0.1, 4, 'delta'), (4, 8, 'theta'),
        (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, 100, 'gamma'), (100, 300, 'fast')
    ]

    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)

    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'std': np.std(epochs, ddof=1, axis=1),
        'iqr': sp_stats.iqr(epochs, axis=1),
        'skew': sp_stats.skew(epochs, axis=1),
        'kurt': sp_stats.kurtosis(epochs, axis=1),
        'nzc': ant.num_zerocross(epochs, axis=1),
        'hmob': hmob,
        'hcomp': hcomp
    }

    # Calculate spectral power features (for EEG + EOG)
    freqs, psd = sp_sig.welch(epochs, sr)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    # Add power ratios for EEG
    # delta = feat['delta']
    # feat['dt'] = delta / feat['theta']
    # feat['ds'] = delta / feat['sigma']
    # feat['db'] = delta / feat['beta']
    # feat['dg'] = delta / feat['gamma']
    # feat['df'] = delta / feat['fast']
    feat['at'] = feat['alpha'] / feat['theta']
    # feat['gt'] = feat['gamma'] / feat['theta']
    # feat['ft'] = feat['fast'] / feat['theta']
    feat['ag'] = feat['gamma'] / feat['alpha']
    # feat['af'] = feat['fast'] / feat['alpha']
    # feat['st'] = feat['sigma'] / feat['theta']
    # feat['bt'] = feat['beta'] / feat['theta']
    # feat['sa'] = feat['sigma'] / feat['alpha']
    # feat['ba'] = feat['beta'] / feat['alpha']
    # feat['sb'] = feat['sigma'] / feat['beta']
    # feat['sg'] = feat['sigma'] / feat['gamma']
    feat['sf'] = feat['sigma'] / feat['fast']
    # feat['bg'] = feat['beta'] / feat['gamma']
    feat['bf'] = feat['beta'] / feat['fast']
    feat['gf'] = feat['gamma'] / feat['fast']


    # Add total power
    idx_broad = np.logical_and(
        freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

    # Calculate entropy and fractal dimension features
    feat['perm'] = np.apply_along_axis(
        ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat['higuchi'] = np.apply_along_axis(
        ant.higuchi_fd, axis=1, arr=epochs)
    feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    # feat.index.name = 'epoch'

    #############################
    # SMOOTHING & NORMALIZATION
    #############################
    roll1 = feat.rolling(window=1, center=True, min_periods=1, win_type='triang').mean()
    roll1[roll1.columns] = robust_scale(roll1, quantile_range=(5, 95))
    roll1 = roll1.iloc[:, 2:].add_suffix('_cmin_norm')

    roll3 = feat.rolling(window=3, center=True, min_periods=1, win_type='triang').mean()
    roll3[roll3.columns] = robust_scale(roll3, quantile_range=(5, 95))
    roll3 = roll3.iloc[:, 2:].add_suffix('_pmin_norm')

    # Add to current set of features
    feat = feat.join(roll1).join(roll3)
    # Remove cols of only zeros
    feat = feat.loc[:, (feat != 0).any(axis=0)]

    return feat

def calc_features_with_delta(epochs, subj):
    # Bandpass filter
    freq_broad = (0.1, 500)
    # FFT & bandpower parameters
    sr = 1000
    bands = [
        (0.5, 4, 'delta'), (4, 8, 'theta'),
        (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, 100, 'gamma'), (100, 300, 'fast')
    ]

    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)

    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'std': np.std(epochs, ddof=1, axis=1),
        'iqr': sp_stats.iqr(epochs, axis=1),
        'skew': sp_stats.skew(epochs, axis=1),
        'kurt': sp_stats.kurtosis(epochs, axis=1),
        'nzc': ant.num_zerocross(epochs, axis=1),
        'hmob': hmob,
        'hcomp': hcomp
    }

    # Calculate spectral power features (for EEG + EOG)
    # calc for 500ms window, it means every two epochs will get the same delta
    double_epochs = np.empty((0, 500))
    for i in range(0, len(epochs), 2):
        if i + 2 < epochs.shape[0]:
            curr = np.concatenate(epochs[i: i + 2]).reshape(1,500)
        else:
            curr = np.concatenate(epochs[i - 1: i + 1]).reshape(1,500)
        double_epochs = np.concatenate((double_epochs, curr))
    freqs, psd = sp_sig.welch(double_epochs, sr, nperseg=500)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
    double_bp = np.repeat(bp, 2, axis=1)
    double_psd = np.repeat(psd, 2, axis=0)
    # Remove the last epoch if the epochs number is odd
    if epochs.shape[0] % 2 != 0:
        double_bp = double_bp[:, :-1]
        double_psd = double_psd[:-1, :]
    for j, (_, _, b) in enumerate(bands):
        feat[b] = double_bp[j]

    # Add power ratios for EEG
    delta = feat['delta']
    feat['dt'] = delta / feat['theta']
    feat['ds'] = delta / feat['sigma']
    feat['db'] = delta / feat['beta']
    feat['dg'] = delta / feat['gamma']
    feat['df'] = delta / feat['fast']
    feat['at'] = feat['alpha'] / feat['theta']
    feat['gt'] = feat['gamma'] / feat['theta']
    feat['ft'] = feat['fast'] / feat['theta']
    feat['ag'] = feat['gamma'] / feat['alpha']
    feat['af'] = feat['fast'] / feat['alpha']

    # Add total power
    idx_broad = np.logical_and(
        freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat['abspow'] = np.trapz(double_psd[:, idx_broad], dx=dx)

    # Calculate entropy and fractal dimension features
    feat['perm'] = np.apply_along_axis(
        ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat['higuchi'] = np.apply_along_axis(
        ant.higuchi_fd, axis=1, arr=epochs)
    feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    # feat.index.name = 'epoch'

    #############################
    # SMOOTHING & NORMALIZATION
    #############################
    roll1 = feat.rolling(window=1, center=True, min_periods=1, win_type='triang').mean()
    roll1[roll1.columns] = robust_scale(roll1, quantile_range=(5, 95))
    roll1 = roll1.iloc[:, 2:].add_suffix('_cmin_norm')

    roll3 = feat.rolling(window=3, center=True, min_periods=1, win_type='triang').mean()
    roll3[roll3.columns] = robust_scale(roll3, quantile_range=(5, 95))
    roll3 = roll3.iloc[:, 2:].add_suffix('_pmin_norm')

    # Add to current set of features
    feat = feat.join(roll1).join(roll3)

    return feat

def run_all_AH(subjects=['38', '396', '398', '402', '406', '415', '416']):
    x_all = np.empty((0, 250))
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    for subj in subjects:
        for channel in ['RAH1', 'LAH1']:
            if not ((subj == '396' and channel == 'RAH1') or (subj == '38' and channel == 'LAH1')):
                x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                features = calc_features(x, subj)
                x_all = np.concatenate((x_all, x))
                y_all = np.concatenate((y_all, y))
                feat_all = pd.concat([feat_all, features], axis=0)
    return x_all, feat_all, y_all

def run_all_multi_channel(subjects=['38', '396', '398', '402', '406', '415', '416']):
    neighbors = {'R': ['RAH1-RAH2', 'RA1'], 'L': ['LAH1-LAH2', 'LA1']}
    x_all = np.empty((0, 250))
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    for subj in subjects:
        for channel in ['RAH1', 'LAH1']:
            if not ((subj == '396' and channel == 'RAH1') or (subj == '38' and channel == 'LAH1')):
                x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                x_all = np.concatenate((x_all, x))
                y_all = np.concatenate((y_all, y))
                features = calc_features(x, subj)
                for neighbor in neighbors[channel[0]]:
                    x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', neighbor)
                    prefix = neighbor.replace(channel[0], '')
                    features_neighbor = calc_features(x, subj).add_prefix(f'{prefix}_')
                    features = pd.concat([features, features_neighbor.iloc[:, 2:]], axis=1)

                feat_all = pd.concat([feat_all, features], axis=0)

    return x_all, feat_all, y_all

def run_all_multi_channel_with_x_neighbor(subjects=['38', '396', '398', '402', '406', '415', '416']):
    neighbors = {'R': ['RAH1-RAH2', 'RA1'], 'L': ['LAH1-LAH2', 'LA1']}
    x_AH = np.empty((0, 250))
    x_A = np.empty((0, 250))
    x_bi = np.empty((0, 250))
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    for subj in subjects:
        for channel in ['RAH1', 'LAH1']:
            if not ((subj == '396' and channel == 'RAH1') or (subj == '38' and channel == 'LAH1')):
                x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                x_AH = np.concatenate((x_AH, x))
                y_all = np.concatenate((y_all, y))
                features = calc_features(x, subj)
                for neighbor in neighbors[channel[0]]:
                    x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', neighbor)
                    if '-' in neighbor:
                        x_bi = np.concatenate((x_bi, x))
                    else:
                        x_A = np.concatenate((x_A, x))
                    prefix = neighbor.replace(channel[0], '')
                    features_neighbor = calc_features(x, subj).add_prefix(f'{prefix}_')
                    features = pd.concat([features, features_neighbor.iloc[:, 2:]], axis=1)

                feat_all = pd.concat([feat_all, features], axis=0)

    # feat_all = feat_all.drop('epoch', axis=1)
    feat_all = feat_all.reset_index(drop=True)
    feat_all.index.name = 'epoch'

    return x_AH, x_bi, x_A, feat_all, y_all