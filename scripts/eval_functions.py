import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from scipy import interp
from itertools import cycle
sns.set_style('white')
import sys

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, \
                            f1_score, log_loss, auc, roc_curve
from sklearn.preprocessing import StandardScaler


def get_datasets(train, val, test, features, scale=False):
    """Return datasets with selected features. Datasets can optional be scaled
    as well."""
    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    # Always scale after splitting data to avoid data leakage
    if scale:
        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_val_z = scaler.fit_transform(X_val)
        X_test_z = scaler.fit_transform(X_test)
        return X_train_z, X_val_z, X_test_z

    else:
        return X_train, X_val, X_test

def corr_matrix(data):
    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corr = data.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10,8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220,10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0,
                square=True, linewidths=.4, annot=True, cbar_kws={'shrink':0.6})

    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)

def one_hot(labels, num_classes=3):
    return np.squeeze(np.eye(num_classes)[labels.reshape(-1)])

def class_label_arr(y):
    """
    One vs. All labels for inputted 3 class multilabel array.
    params:
    y: 1-dimensional label vector
    A one hot encoded label vector is created and divided in a 1-vs-rest manner for each class.
    Returns a one dimensional label array for each class.
    """
    c1_label = one_hot(y)[:, :1]
    c2_label = one_hot(y)[:, 1:2]
    c3_label = one_hot(y)[:, -1:]
    return c1_label, c2_label, c3_label

def one_v_all_arr(arr):
    """
    Return one vs all array with two columns. The first column is
    the sum of the non-relevant classes. The second column is the
    relevant class.

    Parameters
    ----------
    arr : np.array, shape=(n_samples, [n_classes])
          NumPy array with values for each class in separate columns.
    """
    c1 = arr[:, :1]
    c2 = arr[:, 1:2]
    c3 = arr[:, -1:]

    c1_v_all = np.hstack([(c2 + c3), c1])
    c2_v_all = np.hstack([(c1 + c3), c2])
    c3_v_all = np.hstack([(c1 + c2), c3])
    return c1_v_all, c2_v_all, c3_v_all

# CALIBRATION FUNCTIONS
def brier_multi(targets=None, probs=None, num_classes=3):
    onehot_targets = one_hot(targets, num_classes)
    return np.mean(np.sum((probs - onehot_targets)**2, axis=1))

def multiclass_roc_curve(probs, y):
    """ Three class roc curve. """
    y_onehot = one_hot(y)

    n_classes = 3
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(list(y_onehot[:, i]), probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    class_labels = ['Negative', 'Positive', 'Mixed']

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color, label in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(label, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm, classes, figsize=(5,5),
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    np.seterr(divide='ignore', invalid='ignore')
    # Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    length = len(classes)
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=13)
    plt.colorbar()
    tick_marks = np.arange(length)
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    plt.ylim([(length - .5), -.5])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)



def get_ratios_multiclass(probs, y):
    ratios_df = pd.DataFrame(columns=['label', 'pred_neg', 'pred_pos',
                                      'pred_mix', 'pos_neg', 'pos_mix',
                                      'neg_pos', 'neg_mix', 'mix_pos',
                                      'mix_neg'])
    ratios = []
    for probas in probs:

        pos_neg = probas[1]/probas[0]
        pos_mix = probas[1]/probas[2]
        neg_pos = probas[0]/probas[1]
        neg_mix = probas[0]/probas[2]
        mix_pos = probas[2]/probas[1]
        mix_neg = probas[2]/probas[0]

        ratios.append([pos_neg, pos_mix, neg_pos, neg_mix, mix_pos, mix_neg])
    assert len(ratios) == len(probs) == len(y)

    ratios_df['label'] = y.copy()
    ratios_df['pred_neg'] = [x[0] for x in probs]
    ratios_df['pred_pos'] = [x[1] for x in probs]
    ratios_df['pred_mix'] = [x[2] for x in probs]

    ratios_df['pos_neg'] = [x[0] for x in ratios]
    ratios_df['pos_mix'] = [x[1] for x in ratios]
    ratios_df['neg_pos'] = [x[2] for x in ratios]
    ratios_df['neg_mix'] = [x[3] for x in ratios]
    ratios_df['mix_pos'] = [x[4] for x in ratios]
    ratios_df['mix_neg'] = [x[5] for x in ratios]

    return ratios_df

def plot_cm_best_estimator(gs, X_test, y_test, classes):
    preds = gs.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, classes=classes, normalize=True)

def parse_results(results, wvec_name):
    df = pd.DataFrame()

    df['nfeatures'] = np.arange(10000, 100001, 10000)

    col = 'param_ngrams__' + wvec_name + '__ngram_range'

    lg = results[results.param_clf.astype(str).str.contains('Logistic')]
    df['LogReg_Bi'] = lg.mean_test_f1[lg[col]==(1,2)].tolist()
    df['Logreg_Tri'] = lg.mean_test_f1[lg[col]==(1,3)].tolist()
    df['LogReg_Quad'] = lg.mean_test_f1[lg[col]==(1,4)].tolist()

    xgb = results[results.param_clf.astype(str).str.contains('XGB')]
    df['XGB_Bi'] = xgb.mean_test_f1[xgb[col]==(1,2)].tolist()
    df['XGB_Tri'] = xgb.mean_test_f1[xgb[col]==(1,3)].tolist()
    df['XGB_Quad'] = xgb.mean_test_f1[xgb[col]==(1,4)].tolist()

    return df

def plot_ngrams(results, title=None, divide_colors=False, wvec_name=None):

    from bokeh.palettes import Viridis11, Spectral11, Paired, brewer, d3, all_palettes

    df = parse_results(results, wvec_name)
    plt.figure(figsize=(8,6))
    numlines=len(df.columns) - 1

    if divide_colors & (numlines%2 == 0):
        length = int(numlines/2)
        mypalette=all_palettes['Category20'][length]
        lines = ['-',':']*length

        for line, color, col in zip(lines, mypalette*2, df.columns[1:]):
            plt.plot(df.nfeatures, df[col].values, label=col,
                     color=color, linestyle=line)

    else:
        mypalette=all_palettes['Category20'][numlines]
        for color, col in zip(mypalette, df.columns[1:]):
            plt.plot(df.nfeatures, df[col].values, label=col,
                     color=color)

    plt.title(title, fontsize=14)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Validation Set F1", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0);
