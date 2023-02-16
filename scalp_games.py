import numpy as np
import pandas as pd
import mne
from scalp_utils import *
from visbrain.io.rw_hypno import read_hypno
from depth_utils import get_metrics, calc_features_before_split, calc_features_after_split
import joblib
from IPython.display import clear_output
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

sr = 1000
scalp_edf_path = 'C:\\Users\\user\\PycharmProjects\\pythonProject\\%s_clean.edf'


def remove_rem(raw, edf):
    hypno_file = edf.split('_')[0] + '_hypno.txt'

    # read hypnogram old format (1sec)
    hypno, sf_hypno = read_hypno(hypno_file, time=None, datafile=None)

    # make raw object into epochs (30 sec)
    dummy_events = mne.make_fixed_length_events(raw, id=1, duration=30)[:len(hypno)]

    # incorporate the scoring into the events file:
    dummy_events[:, 2] = hypno
    event_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'art': -1}

    # epoch data into 30sec pieces:
    epochs = mne.Epochs(raw, events=dummy_events, event_id=event_dict, tmin=0,
                        tmax=30, baseline=(0, 0), on_missing='ignore')
    epochs.drop(epochs['REM'].selection)
    nrem_raw = mne.io.RawArray(np.concatenate(epochs.get_data(), axis=1), raw.info)

    return nrem_raw


def get_all_y_multi_channel_side(side, subjects=['38', '396', '398', '402', '406', '415', '416']):
    neighbors = {'R': ['RAH1-RAH2', 'RA1'], 'L': ['LAH1-LAH2', 'LA1']}
    y_all = np.empty(0)
    for subj in subjects:
        channel = side + 'AH1'
        if not ((subj == '396' and channel == 'RAH1') or (subj == '38' and channel == 'LAH1')):
            x = format_raw_night(scalp_edf_path % subj, channel)
            features = calc_features(x, subj)
            for neighbor in neighbors[channel[0]]:
                x_neighbor = format_raw_night(scalp_edf_path % subj, neighbor)
                prefix = neighbor.replace(channel[0], '')
                features_neighbor = calc_features(x_neighbor, subj).add_prefix(f'{prefix}_')
                features = pd.concat([features, features_neighbor.iloc[:, 2:]], axis=1)

            side1_y = model.predict(features[features_names[1:]])
            y_all = np.concatenate((y_all, side1_y))

    return y_all


def oversampling_smote(feat, y):
    cls_over_loo = {
        "Random Forest": RandomForestClassifier(),
        "LGBM": LGBMClassifier(),
    }

    f, axes = plt.subplots(1, len(cls_over_loo), figsize=(10, 5), sharey='row')

    oversample = SMOTE()
    x_over, y_over = oversample.fit_resample(feat, y)
    X_train, X_test, y_train, y_test = train_test_split(x_over, y_over, stratify=y_over, random_state=20)
    X_train_data = X_train.iloc[:, 2:]
    selector = VarianceThreshold(.1)
    selector.fit_transform(X_train_data)
    X_train_data = X_train_data[X_train_data.columns[selector.get_support(indices=True)]]
    X_test_data = X_test[X_train_data.columns]

    for i, (key, classifier) in enumerate(cls_over_loo.items()):
        y_pred = classifier.fit(X_train_data, y_train).predict(X_test_data)
        cf_matrix = confusion_matrix(y_test, y_pred)
        metrics = get_metrics(cf_matrix)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)
        text = """precision: {0}
                  recall: {1}""".format(str('{0:.2f}'.format(metrics['precision'])), str('{0:.2f}'.format(metrics['recall'])))
        axes[i].annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                         xytext=(-60, -40), textcoords='offset points',
                         ha='right', va='top')
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')
            y_lgbm = y_pred
        else:
            y_rf = y_pred

    f.text(0.45, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    plt.show()


def loo_top_confidence(cls, feat_out, y_out, top_num=30):
    f, axes = plt.subplots(1, len(cls), figsize=(10, 5), sharey='row')

    feat_396_data = feat_out.iloc[:, 2:]
    feat_396_data = feat_396_data[cls['LGBM'].feature_name_]

    for i, (key, classifier) in enumerate(cls.items()):
        y_pred = classifier.predict_proba(feat_396_data).T
        prob = np.sort(y_pred[1])[::-1].T[top_num]
        cf_matrix = confusion_matrix(y_out,  [p >= prob for p in y_pred[1]])
        metrics = get_metrics(cf_matrix)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)
        precision = '{0:.2f}'.format(metrics['precision'])
        recall = '{0:.2f}'.format(metrics['recall'])
        text = """precision: {0}
                      recall: {1}""".format(precision, recall)
        axes[i].annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                         xytext=(-60, -40), textcoords='offset points',
                         ha='right', va='top')
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')

    f.text(0.45, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()


def permutation_test(y_pred, y_test, save=True):
    random_metrics = []
    for i in range(0, 100000):
        np.random.shuffle(y_pred)
        cf_matrix = confusion_matrix(y_test, y_pred)
        random_metrics.append(get_metrics(cf_matrix))

    if save:
        joblib.dump(random_metrics, 'shuffle_metrics.pkl')

    return random_metrics


def plot_permutation():
    random_metrics = joblib.load('shuffle_metrics.pkl')
    precision_random = [x['precision'] for x in random_metrics]
    plt.rcParams.update({'font.size': 12, 'font.weight': 'medium'})
    x = np.array(precision_random)
    plt.hist(precision_random, bins=15, edgecolor="black")
    plt.xlim(0, 1)
    plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1.5)
    plt.axvline(x.max(), color='green', linestyle='dashed', linewidth=1.5)
    plt.axvline(0.9, color='black', linestyle='dashed', linewidth=1.5)
    plt.xlabel('Precision', fontsize=14)

    # min_ylim, max_ylim = plt.ylim()
    # plt.text(x.mean()*1.2, max_ylim*0.9, 'Mean: {:.2f}%'.format(x.mean() * 100))
    # plt.text(x.max()*1.1, max_ylim*0.7, 'Max: {:.2f}%'.format(x.max() * 100))
    # plt.text(x.max()*4, max_ylim*0.7, 'classifier: {:.2f}%'.format(90))


def spikes_per_stage(subj):
    subj = '416'
    filename = f'C:\\Users\\user\\PycharmProjects\\pythonProject\\hypno_{subj}.csv'
    hypno_df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['Stage', 'Time']).drop([0])
    spikes_df = pd.read_csv(f'thesis_depth_lgbm_{subj}.csv')
    spike_per_stage = {'Wake': 0, 'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}
    for spike in spikes_df.values:
        spike_sec = spike[0] * 250 / 1000
        if len(hypno_df[hypno_df['Time'] < spike_sec]) == 0:
            spike_per_stage['Wake'] += 1
        else:
            stage_index = len(hypno_df[hypno_df['Time']< spike_sec])
            stage = hypno_df.iloc[stage_index]['Stage']
            spike_per_stage[stage] += 1

    return spike_per_stage


def plot_results_by_confidence(y_rf, y_lgbm):
    x_ticks = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    # y_rf = [78, 81 ,82 ,86, 88 ,100 ,100 , 100, 100, 100]
    # y_lgbm = [63, 65, 67, 70, 72, 73, 75, 74, 77, 100]
    plt.rcParams.update({'font.size': 14, 'font.weight': 'medium'})
    labels = x_ticks
    tp_rf = [78, 60, 45, 32, 22, 15, 8, 3, 2, 2]
    tp_lgbm = [119, 102, 94, 88, 79, 62, 47, 34, 20, 6]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width/2, tp_rf, width, label='RF', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, tp_lgbm, width, label='LGBM', color='peachpuff')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('True Positive / Precision')
    ax.set_xlabel('Probability')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    # ax.set_xticks(labels)
    ax.legend()

    ax2 = ax.twiny()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax2.plot(x_ticks, y_rf, linewidth=3)
    ax2.plot(x_ticks, y_lgbm, linewidth=3)
    # ax2.spines.right.set_visible(False)
    # ax2.spines.top.set_visible(False)

    fig.tight_layout()


def precision_recall_curve(cls, X_test_data, y_test, y_pred):
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    plt.rcParams.update({'font.size': 14, 'font.weight': 'medium'})
    plt.figure(figsize=(8,5))
    model_lgbm = cls["LGBM"]
    model_rf = cls['Random Forest']
    testX, testy = X_test_data, y_test
    lr_probs = model_lgbm.predict_proba(testX)[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, y_pred), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    plt.plot(lr_recall, lr_precision, label='LGBM (AUC = 0.993)', color='tab:blue')
    lr_probs = model_rf.predict_proba(testX)[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, y_pred), auc(lr_recall, lr_precision)
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    plt.plot(lr_recall, lr_precision, label='RF (AUC = 0.999)', color='tab:orange')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.gca().set_ylim([0.5, 1.05])
    plt.gca().set_xlim([0, 1.05])
    # show the legend
    plt.legend(loc=3, prop={'size': 14})
    # show the plot
    plt.show()
