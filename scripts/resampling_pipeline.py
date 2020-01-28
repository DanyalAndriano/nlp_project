import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import metrics

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours



# -------------------------RESAMPLING PIPELINE ---------------------------------------------------------
def model_resampling_pipeline(X_train, X_test, y_train, y_test, model,
                              b= 0.5, name = '', eval_show=True, columns=None):

    if not hasattr(model, 'predict_proba'):
        model = CalibratedClassifierCV(model, cv=3)
    else:
        model = model

    results = {'ordinary': {},
               'class_weight': {},
               'oversample': {},
               'undersample': {}}

    # ------ No balancing ------
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probas = [x[1] for x in model.predict_proba(X_test)]
    scores = metrics.classification_report(y_test, predictions,
                                           target_names=['negative', 'positive', 'mixed'],
                                           output_dict=True)


    w_precision = scores['macro avg']['precision']
    w_recall = scores['macro avg']['recall']
    w_fscore = scores['macro avg']['f1-score']

    results['ordinary'] = {'w_precision': w_precision, 'w_recall': w_recall, 'w_fscore': w_fscore,
                           'predictions': np.array(predictions), 'probas':probas}


    # ------ Class weight ------
    if 'class_weight' in model.get_params().keys():
        model.set_params(class_weight='balanced')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probas = [x[1] for x in model.predict_proba(X_test)]
        scores = metrics.classification_report(y_test, predictions,
                                               target_names=['negative', 'positive', 'mixed'],
                                               output_dict=True)

        w_precision = scores['macro avg']['precision']
        w_recall = scores['macro avg']['recall']
        w_fscore = scores['macro avg']['f1-score']

        results['class_weight'] = {'w_precision': w_precision, 'w_recall': w_recall, 'w_fscore': w_fscore,
                                   'predictions': np.array(predictions), 'probas':probas}


    # ------------ OVERSAMPLING TECHNIQUES ------------
    techniques = [RandomOverSampler(),
                  SMOTE(),
                  ADASYN()]

    for sampler in techniques:
        technique = sampler.__class__.__name__
        X_resampled, y_resampled = sampler.fit_sample(X_train, y_train)

        X_resampled = pd.DataFrame(X_resampled)
        if columns:
            X_resampled.columns = columns
        else:
            X_resampled.columns = X_train.columns

        model.fit(X_resampled, y_resampled)
        predictions = model.predict(X_test)
        probas = [x[1] for x in model.predict_proba(X_test)]
        scores = metrics.classification_report(y_test, predictions,
                                               target_names=['negative', 'positive', 'mixed'],
                                               output_dict=True)

        w_precision = scores['macro avg']['precision']
        w_recall = scores['macro avg']['recall']
        w_fscore = scores['macro avg']['f1-score']

        results['oversample'][technique] = {'w_precision': w_precision, 'w_recall': w_recall, 'w_fscore': w_fscore,
                                             'predictions': np.array(predictions), 'probas':probas}


    # ------------ UNDERSAMPLING TECHNIQUES ------------
    techniques = [RandomUnderSampler(),
                  NearMiss(version=1),
                  NearMiss(version=2),
                  TomekLinks(),
                  EditedNearestNeighbours()]

    for sampler in techniques:
        technique = sampler.__class__.__name__
        if technique == 'NearMiss': technique+=str(sampler.version)
        X_resampled, y_resampled = sampler.fit_sample(X_train, y_train)

        X_resampled = pd.DataFrame(X_resampled)
        if columns:
            X_resampled.columns = columns
        else:
            X_resampled.columns = X_train.columns

        model.fit(X_resampled, y_resampled)
        predictions = model.predict(X_test)
        probas = [x[1] for x in model.predict_proba(X_test)]
        scores = metrics.classification_report(y_test, predictions,
                                               target_names=['negative', 'positive', 'mixed'],
                                               output_dict=True)

        w_precision = scores['macro avg']['precision']
        w_recall = scores['macro avg']['recall']
        w_fscore = scores['macro avg']['f1-score']

        results['undersample'][technique] = {'w_precision': w_precision, 'w_recall': w_recall, 'w_fscore': w_fscore,
                                             'predictions': np.array(predictions), 'probas':probas}

    if eval_show:
        evaluate_method(results, y_test, 'undersample', title = name + '\nUndersampled')
        evaluate_method(results, y_test, 'oversample', title = 'Oversampled')

    return results

def evaluate_method(results, y, sampling, labels=['positive', 'negative', 'mixed'], title=None):
    plt.style.use('seaborn-white')

    ordinary_preds = results['ordinary']['predictions']
    ordinary_dict = metrics.classification_report(y, ordinary_preds,
                                                  target_names=['negative', 'positive', 'mixed'],
                                                  output_dict=True)


    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(11, 5))

    for i, label in enumerate(labels):
        ax[i].axhline(ordinary_dict[label]['recall'],
                      label='No Resampling (Macro F1: {:.3f})'.format(ordinary_dict['macro avg']['f1-score']))

        if results['class_weight']:
            class_weight_preds = results['class_weight']['predictions']
            class_dict = metrics.classification_report(y, class_weight_preds,
                                                  target_names=['negative', 'positive', 'mixed'],
                                                  output_dict=True)

            ax[i].bar(0, class_dict[label]['recall'],
                      label='Adjust Class Weight (Macro F1: {:.3f})'.format(class_dict['macro avg']['f1-score']))

        for j, (technique, result) in enumerate(results[sampling].items()):
            preds = result['predictions']
            tech_dict = metrics.classification_report(y, preds,
                                                      target_names=['negative', 'positive', 'mixed'],
                                                      output_dict=True)

            ax[i].bar(j+1, tech_dict[label]['recall'],
                      label='{} (Macro F1: {:.3f})'.format(technique, tech_dict['macro avg']['f1-score']))

        ax[i].set_title(f'Recall\n{label.upper()}')
        ax[i].set_xticklabels(labels=[])

    fig.suptitle(title, y=1.1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               fancybox=True, shadow=True, prop={'size': 12})


def get_report(results, y, method, sampling=None, print_report=True):
    if sampling:
        preds = results[method][sampling]['predictions']
    else:
        preds = results[method]['predictions']
    edited_report = metrics.classification_report(y, preds,
                                       target_names=['negative', 'positive', 'mixed'])
    if print_report:
        print(edited_report)
    return edited_report
