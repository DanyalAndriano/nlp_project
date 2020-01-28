import calibration
import calibration.stats as stats
import calibration.binning as binning
import netcal
import numpy as np
import pandas as pd
import scipy
import sklearn
from netcal.binning import IsotonicRegression
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, f1_score

from calibration_functions import brier_multi, calc_classwise_ece, classwise_diagrams, \
                                  confidence_diagram, one_hot
from eval_functions import plot_confusion_matrix

class Evaluation:
    """
    The evaluation class inherits parameters from the calibration class object.

    calibrated: np.ndarray, shape=(n_samples, 3)
        Estimated confidences (probabilities) from classifier.
    labels: np.ndarray, shape=(n_samples,)
        NumPy 1-D array with ground truth labels.
    method: str, Calibration method used.
    """
    def __init__(self,
                 calibrated,
                 labels,
                 method):
        self.calibrated = calibrated
        self.labels = labels
        self.method = method

    def diagrams(self, n_bins=10, confidence=True, classwise=False):
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

        Returns:
            Reliability diagrams.
        """
        if confidence:
            print('Confidence-Reliability Visual Evaluation')
            print('-'*60)
            confidence_diagram(self.calibrated, self.labels, n_bins)
        if classwise:
            print('Classwise-Reliability Visual Evaluation')
            print('-'*60)
            classwise_diagrams(self.calibrated, self.labels, self.method, n_bins)

    def evaluation_metrics(self, n_bins=10, verbose=True):
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

        Returns:
            Dataframe with all evaluation metrics.
        """
        ece = ECE(n_bins)
        ece_score = ece.measure(self.calibrated, self.labels)
        classwise_ece = calc_classwise_ece(self.calibrated, self.labels)
        dd_ece = stats.ece(self.calibrated, one_hot( self.labels), binning=binning.DataDependentBinning())
        brier = brier_multi(self.labels, self.calibrated)
        nll = log_loss(self.labels, self.calibrated)
        f1 = f1_score(self.labels, np.argmax(self.calibrated, axis=1), average="macro")

        df = pd.DataFrame(columns=['ECE', 'Classwise ECE', 'Adaptive ECE', 'Brier', 'Neg Log-Likelihood', 'F1-Macro'])
        df.loc[0] = ece_score, classwise_ece, dd_ece, brier, nll, f1

        if verbose:
            print(self.method + ' - Calibration Metrics')
            print('-'*50)
            print('ECE: ', round(ece_score, 4))
            print('Classwise/ Static ECE: ', round(classwise_ece, 4))
            print('Adaptive ECE: ', round(dd_ece, 4))
            print('Brier Multi Score: ', round(brier, 4))
            print('f1 - macro: ', round(f1, 4))
            print('Negative Log-Likelihood: ', round(nll, 4))
        return df

    def get_confusion_matrix(self, classes, normalize=True, title=None):
        preds = np.argmax(self.calibrated, axis=1)
        cm = confusion_matrix(self.labels, preds)
        plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix')

class DirichletScaling(Evaluation):
    """
    Initialize class.

    Parameters:
    ---------------------
        y_test: np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels for test sample
            (i.e., not sample used to fit model).
        calibrated: np.ndarray, shape=(n_samples, 3)
            Estimated confidences (probabilities) from classifier.
        method: str, default: 'Dirichlet Scaling'
    """

    def __init__(self,
                 y_test,
                 calibrated=None,
                 method='Dirichlet Scaling'):
        super().__init__(calibrated=calibrated,
                         labels=y_test,
                         method=method)

        self.calibrated = calibrated
        self.labels = y_test
        self.method = method
        self.logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg')

    def fit(self, train, y_train):
        """
        Fit multinomial logistic regression model on probability outputs
        from classifier. Do not use the outputs from the original training dataset.
        Probability estimates from the validation set are ideal.

        Params:
        --------
            train: np.ndarray, shape=(n_samples, [n_classes])
                Probability inputs to train calibration model.
            y_train: np.ndarray, shape=(n_samples,)
                NumPy 1-D array with ground truth labels for train sample
        """
        # inputs are the log transformed probabilities.
        self.logreg.fit(np.log(train), y_train)

    def transform(self, test):
        """
        Use probability estimates from the unseen test set to evaluate calibration.

        Returns:
            Calibrated probabilities for the test set.
        """
        self.calibrated = self.logreg.predict_proba(np.log(test))
        return self.calibrated

class TemperatureScaling(Evaluation):
    """
    Initialize class.

    Parameters:
    ------------
        temp (float): starting temperature, default 1
        maxiter (int): maximum iterations done by optimizer,
            however 8 iterations have been maximum.
        solver: default: "BFGS"
        calibrated: np.ndarray, shape=(n_samples, 3)
            Estimated confidences (probabilities) from classifier.
        labels: np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels for test sample
        method: str, default: 'Temperature Scaling'
    """
    def __init__(self,
                 temp=1,
                 maxiter=50,
                 solver="BFGS",
                 calibrated=None,
                 labels=None,
                 method='Temperature Scaling'):
        super().__init__(calibrated=calibrated,
                         labels=labels,
                         method=method)

        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
        self.calibrated = calibrated
        self.labels = labels
        self.method = method

    def _loss_fun(self, x, probs, labels):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self._predict(probs, x)
        loss = log_loss(y_true=labels, y_pred=scaled_probs)
        return loss

    def _predict(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities.

        Params:
        --------
        logits: logits values of data (output from neural network)
            for each class (shape [samples, classes])
        temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [n_samples, classes])
        """
        if not temp:
            return _softmax(logits/self.temp)
        else:
            return _softmax(logits/temp)

    # Find the temperature-paramter to rescale logit inputs to softmax function
    def fit(self, logits, labels, num_classes=3):
        """
        Trains the model and finds the optimal temperature-parameter w.r.t. NLL,
        using the logits as inputs

        Params:
        ----------
            logits: the output from neural network for each class (shape [n_samples, classes])
            labels: a list of ground truth class labels

        Returns:
            the results of optimizer after minimizing is finished.
        """
        self.labels = labels
        onehot_labels = one_hot(self.labels, num_classes=num_classes)
        opt = minimize(self._loss_fun, x0=1, args=(logits, onehot_labels),
                       options={'maxiter':self.maxiter}, method=self.solver)
        self.temp = opt.x[0]
        return opt

    def transform(self, logits):
        """
        Use logit estimates to evaluate calibration.

        Returns:
            Vector of calibrated probabilities, size=(n_samples, [n_classes]).
        """
        self.calibrated = _softmax(logits/self.temp)
        return self.calibrated

class Isotronic(Evaluation):
    """
    Isotonic Regression is a calibration method to correct any monotonic distortion.
    """

    def __init__(self,
                 labels=None,
                 calibrated=None,
                 method='Isotronic Regression',
                 independent_probabilities=False):
        super().__init__(calibrated=calibrated,
                         labels=labels,
                         method=method)
        self.calibrated = calibrated
        self.labels = labels
        self.method = method
        self.iso = IsotonicRegression(independent_probabilities)

    def fit_transform(self, train, y_train):
        self.labels = y_train
        self.calibrated = self.iso.fit_transform(train, y_train)
        return self.calibrated

class Uncalibrated(Evaluation):
    """
    Initialize class.

    Parameters:
    ------------
        probs: np.ndarray, shape=(n_samples, 3)
            Estimated confidences (probabilities) from classifier.
        labels: np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels for test sample
            (i.e., not sample used to fit model).
        method: str, default: 'Uncalibrated'
    """
    def __init__(self,
                 probs,
                 labels,
                 method='Uncalibrated'):
        super().__init__(calibrated=probs,
                         labels=labels,
                         method=method)
        self.calibrated = probs
        self.labels = labels
        self.method = method

def _softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)
