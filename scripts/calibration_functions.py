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

from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import calibration
import calibration.stats as stats
import calibration.binning as binning

import netcal
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from netcal.binning import IsotonicRegression

# -------CALIBRATION FUNCTIONS------------------------------------------------
def brier_multi(targets=None, probs=None, num_classes=3):
    """Calculate Breir loss for multiple classes."""
    onehot_targets = one_hot(targets, num_classes)
    return np.mean(np.sum((probs - onehot_targets)**2, axis=1))

def calc_classwise_ece(probs, y, n_bins=10):
    """
    Calculate ECE (estimated calibration error) for each
    sentiment class and returns the average.

    Parameters:
    ------------------
    probs: np.ndarray, shape=(n_samples, 3)
        Estimated confidences (probabilities) from classifier.
    y: np.ndarray, shape=(n_samples,)
        NumPy 1-D array with ground truth labels.
    n_bins: n_bins: int, default: 10
        Discretize the probability interval into a fixed number of bins
        and assign predicted probabilities to each bin.

    Return:
        Classwise-ECE.
    """
    c1_probs = probs[:, :1]
    c2_probs = probs[:, 1:2]
    c3_probs = probs[:, -1:]

    c1_label = one_hot(y)[:, :1]
    c2_label = one_hot(y)[:, 1:2]
    c3_label = one_hot(y)[:, -1:]

    ece = ECE(n_bins)
    c1_ece = ece.measure(c1_probs, c1_label)
    c2_ece = ece.measure(c2_probs, c2_label)
    c3_ece = ece.measure(c3_probs, c3_label)

    return (c1_ece + c2_ece + c3_ece)/3

def classwise_diagrams(probs, labels, method, n_bins):
    """
    Plot classwise-reliability diagrams for sentiment classes.

    Parameters:
    ----------------
    probs: np.ndarray, shape=(n_samples, 3)
        Estimated confidences (probabilities) from classifier.
    labels: np.ndarray, shape=(n_samples,)
        NumPy 1-D array with ground truth labels.
    method: str, Calibration method used.
    n_bins: n_bins: int, default: 10
        Discretize the probability interval into a fixed number of bins
        and assign predicted probabilities to each bin.

    Return:
        Classwise-reliability diagram for each class.
    """
    y_c1, y_c2, y_c3 = class_label_arr(labels)
    c1_v_all, c2_v_all, c3_v_all = one_v_all_arr(probs)
    plot_classwise_reliability(c1_v_all, y_c1, n_bins, title_suffix = 'Negative Class ' + method)
    plot_classwise_reliability(c2_v_all, y_c2, n_bins, title_suffix = 'Positive Class ' + method)
    plot_classwise_reliability(c3_v_all, y_c3, n_bins, title_suffix = 'Mixed Class ' + method)

def class_label_arr(y):
    """
    One vs. All labels for inputted 3 class multilabel array.

    Parameters:
    ---------
    y: np.ndarray, shape=(n_samples,)
       NumPy 1-D array with ground truth labels.

    Return:
        One hot encoded label vector divided in a 1-vs-rest manner for each class.
    """
    c1_label = one_hot(y)[:, :1]
    c2_label = one_hot(y)[:, 1:2]
    c3_label = one_hot(y)[:, -1:]
    return c1_label, c2_label, c3_label

def confidence_diagram(probs, labels, n_bins):
    """
    Plot overall confidence-reliability diagrams for classifier estimates.

    Parameters:
    ----------------
    probs: np.ndarray, shape=(n_samples, 3)
        Estimated confidences (probabilities) from classifier.
    labels: np.ndarray, shape=(n_samples,)
        NumPy 1-D array with ground truth labels.
    n_bins: n_bins: int, default: 10
        Discretize the probability interval into a fixed number of bins
        and assign predicted probabilities to each bin.

    Return:
        Probability histogram and confidence-reliability diagram.
    """
    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(probs, labels)

def eval_metrics(probs, y, method, n_bins=10, verbose=True):
    """
    Calculates proper losses, calibration error metrics, and the
    macro weighted F1-score (to evaluate predictive performance).

    Expected Calibration Error: Discretize the probability interval into a
        fixed number of bins and assign predicted probabilities to each bin.
        The calibration error is the difference between the number of correct
        probabilities (accuracy) and the mean of all probabilities (confidence)
        for each bin.
    Classwise ECE: The ECE calculated for each class.
    Adaptive ECE: The Adaptive ECE focuses on those bins where predictions are
        made rather than weighing all bins equally. This metric spaces the bin
        intervals in such a way that each contains an equal number of predictions.
    Brier Score: "The Brier score measures the mean squared difference between
        (1) the predicted probability assigned to the possible outcomes for item i,
        and (2) the actual outcome. Therefore, the lower the Brier score is for a
        set of predictions, the better the predictions are calibrated." sklearn metrics
    Negative Log-Likelihood: The NLL also average the error on every single
        instance to calculate the calibration error.
    F1-Macro weighted: The weighted average of the precision and recall.
        Calculate metrics for each label, and find their unweighted mean.

    Parameters:
    ----------------
    probs: np.ndarray, shape=(n_samples, 3)
        Estimated confidences (probabilities) from classifier.
    y: np.ndarray, shape=(n_samples,)
        NumPy 1-D array with ground truth labels.
    method: str, Calibration method used.
    n_bins: n_bins: int, default: 10
        Discretize the probability interval into a fixed number of bins
        and assign predicted probabilities to each bin.
    verbose: bool, default: True
        Print metrics as output.

    Return:
        Dataframe with all evaluation metrics.
    """
    ece = ECE(n_bins)
    ece_score = ece.measure(probs, y)
    classwise_ece = calc_classwise_ece(probs, y)
    dd_ece = stats.ece(probs, one_hot(y), binning=binning.DataDependentBinning())
    brier = brier_multi(y, probs)
    nll = log_loss(y, probs)
    f1 = f1_score(y, np.argmax(probs, axis=1), average="macro")

    df = pd.DataFrame(columns=['ECE', 'Classwise ECE', 'Adaptive ECE', 'Brier',
                               'Neg Log-Likelihood', 'F1-Macro'])
    df.loc[0] = ece_score, classwise_ece, dd_ece, brier, nll, f1

    if verbose:
        print(method + ' - Calibration Metrics')
        print('-'*50)
        print('ECE: ', round(ece_score, 4))
        print('Classwise/ Static ECE: ', round(classwise_ece, 4))
        print('Adaptive ECE: ', round(dd_ece, 4))
        print('Brier Multi Score: ', round(brier, 4))
        print('f1 - macro: ', round(f1, 4))
        print('Negative Log-Likelihood: ', round(nll, 4))
    return df

def one_hot(labels, num_classes=3):
    """ Returns a one-hot encoded label vector."""
    return np.squeeze(np.eye(num_classes)[labels.reshape(-1)])

def one_v_all_arr(arr):
    """
    Creates a one-vs-all array with two columns. The first column is
    the sum of the non-relevant classes. The second column is the
    relevant class.

    Parameters
    ----------
    arr : np.ndarray, shape=(n_samples, 3)
          NumPy array with values for each class in separate columns.

    Return:
        One-vs-all array for each sentiment class.
    """
    c1 = arr[:, :1]
    c2 = arr[:, 1:2]
    c3 = arr[:, -1:]

    c1_v_all = np.hstack([(c2 + c3), c1])
    c2_v_all = np.hstack([(c1 + c3), c2])
    c3_v_all = np.hstack([(c1 + c2), c3])
    return c1_v_all, c2_v_all, c3_v_all

def plot_classwise_reliability(X: np.ndarray, y: np.ndarray, n_bins=10,\
                               title_suffix: str = None):
    """
    Plot classwise-reliability diagrams to visualize miscalibration within
    each class.

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, [n_classes])
        NumPy array with confidence values for each prediction.
        1-D for binary classification, 2-D for multi class (softmax).
    y : np.ndarray, shape=(n_samples,)
        NumPy 1-D array with ground truth labels.
    n_bins: n_bins: int, default: 10
        Discretize the probability interval into a fixed number of bins
        and assign predicted probabilities to each bin.
    title_suffix : str, optional, default: None
        Suffix for plot title.

    Return:
        Confidence reliability diagram for each class.
    """
    plt.figure(figsize=(6, 4))

    bins = n_bins
    num_samples = y.size
    y = np.squeeze(y)

    title = "Classwise-Reliability Diagram"

    # -----------------------------------------
    # get prediction labels and sort arrays
    if len(X.shape) == 1:

        # first, get right binary predictions (y=0 or y=1)
        predictions = np.where(X > 0.5, 1, 0)

        # calculate average accuracy and average confidence
        total_accuracy = np.equal(predictions, y).sum() / num_samples

        prediction_confidence = np.where(X > 0.5, X, 1. - X)
        total_confidence = np.sum(prediction_confidence) / num_samples

        # plot confidence estimates only for y=1
        # thus, set predictions to 1 for each sample
        predictions = np.ones_like(X)
        title += " for y=1"

    else:
        predictions = np.argmax(X, axis=1)
        X = np.max(X[:, -1:], axis=1)

        # calculate average accuracy and average confidence
        total_accuracy = np.equal(predictions, y).sum() / num_samples
        total_confidence = np.sum(X) / num_samples

    # -----------------------------------------

    # prepare visualization metrics
    bin_confidence = np.zeros(bins)
    bin_frequency = np.zeros(bins)
    bin_gap_freq = np.zeros(bins)
    bin_samples = np.zeros(bins)
    bin_color = ['blue', ] * bins

    # iterate over bins, get avg accuracy and frequency of each bin and add to ECE
    for i in range(1, bins+1):

        # get lower and upper boundaries
        low = (i - 1) / float(bins)
        high = i / float(bins)
        condition = (X > low) & (X <= high)

        num_samples_bin = condition.sum()
        if num_samples_bin <= 0:
            bin_confidence[i-1] = bin_frequency[i-1] = (high + low) / 2.0
            bin_color[i-1] = "yellow"
            continue

        # cal normalized frequency
        avg_frequency = round(y[condition].sum() / num_samples_bin, 2)

        bin_confidence[i-1] = (high + low) / 2.0
        bin_frequency[i-1] = avg_frequency
        bin_gap_freq[i-1] = bin_confidence[i-1] - avg_frequency
        bin_samples[i-1] = num_samples_bin

    bin_samples /= num_samples

    # -----------------------------------------
    # plot stuff
    if title_suffix is not None:
        plt.title('Classwise-Reliability Diagram - ' + title_suffix)
    else:
        plt.title('Classwise-Reliability Diagram')

        plt.bar(bin_confidence, height=bin_frequency, width=1./bins, align='center', edgecolor='black')
        plt.bar(bin_confidence, height=bin_gap_freq, bottom=bin_frequency, width=1./bins, align='center',
                edgecolor='black', color='red', alpha=0.6)

        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))

        plt.xlabel('Confidence')
        plt.ylabel('Frequency')

        plt.legend(['Perfect Calibration', 'Observed Frequency', 'Gap'])
        plt.tight_layout()

        plt.show()

def reliability_diagrams(probs, labels, method, n_bins=10, confidence=True,
                         classwise=False):
    """Plot confidence and classwise reliability diagrams to evaluate
    calibration.

    Params:
    ---------
    n_bins: int, default: 10
        Discretize the probability interval into a fixed number of bins
        and assign predicted probabilities to each bin. The calibration
        error is the difference between the number of correct probabilities
        (accuracy for overall, frequency for classwise) and the mean of
        all probabilities (confidence) for each bin.
    confidence: bool, default: True
        Plot overall reliability for estimated confidences.
    classwise : bool, default: False
        Plot classwise-reliability diagrams - one per class.

    Returns reliability diagrams.
    """
    if confidence:
        print('Confidence-Reliability Visual Evaluation')
        print('-'*60)
        confidence_diagram(probs, labels, n_bins)
    if classwise:
        print('Classwise-Reliability Visual Evaluation')
        print('-'*60)
        classwise_diagrams(probs, labels, method, n_bins)
