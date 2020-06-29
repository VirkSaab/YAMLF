import sklearn.metrics as skm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__KNOWN_METRICS__ = [
    'acc', 'auc', 'mcc', 'tp_tn_fp_fn', 'ppv', 'tpr', 'fpr', 'f1',
    'sensitivity', 'specificity', 'prevalence', 'confusion_matrix'
    'dice_coef', 'soft_dice_loss', 
    ]

class SKMetrics:
    f"""Evaluation metrics class. Available metrics {__KNOWN_METRICS__}"""
    def __init__(self, metrics:list):
        self.metrics = metrics
        self.apply = self.with_probs if 'auc' in self.metrics else self.without_probs

    def without_probs(self, y_true, y_pred, y_probs=None):
        return {met:getattr(self, met)(y_true=y_true, y_pred=y_pred) for met in self.metrics}

    def with_probs(self, y_true, y_pred, y_probs):
        if y_probs is None: raise ValueError('y_probs is required for AUC.')
        _auc = self.auc(y_true=y_true, y_probs=y_probs)
        _metrics = self.metrics.copy()
        _metrics.remove('auc')
        results = {met:getattr(self, met)(y_true=y_true, y_pred=y_pred) for met in _metrics}
        results['auc'] = _auc
        return results

    def __call__(self, y_true, y_pred, y_probs=None):
        return self.apply(y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    
    @staticmethod
    def acc(y_true, y_pred): return skm.accuracy_score(y_true, y_pred)
    @staticmethod
    def auc(y_true, y_probs): return skm.roc_auc_score(y_true, y_score=y_probs)
    @staticmethod
    def mcc(y_true, y_pred): return skm.matthews_corrcoef(y_true, y_pred)
    @staticmethod
    def f1(y_true, y_pred): return skm.f1_score(y_true, y_pred)
    @staticmethod
    def recall(y_true, y_pred): return skm.recall_score(y_true, y_pred)
    @staticmethod
    def precision(y_true, y_pred): return skm.precision_score(y_true, y_pred)
    @staticmethod
    def confusion_matrix(y_true, y_pred): return skm.confusion_matrix(y_true, y_pred).ravel()


class Metrics(SKMetrics):
    def __init__(self, metrics:list):
        super().__init__(metrics)
    
    @staticmethod
    def tp_tn_fp_fn(y_true, y_pred, threshold=0.5):
        if any(y_true > 1):
            raise NotImplementedError("Use this for multiclass. https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py")
        y_pred = y_pred >= threshold
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}

    @staticmethod
    def acc(y_true, y_pred, y_probs=None, threshold=0.5):
        """Simple accuracy metric"""
        print(y_true, y_pred)
        m = Metrics.tp_tn_fp_fn(y_true, y_pred, threshold)
        TP, TN, FP, FN = m['TP'], m['TN'], m['FP'], m['FN']
        print("hola", TP, TN, FP, FN)
        raise ValueError
        return np.nanmean((TP + TN) / (TP + TN + FP + FN))

    @staticmethod
    def prevalence(y_true, **kwargs):
        r"""
        Another important concept is prevalence. 
        * In a medical context, prevalence is the proportion of people
          in the population who have the disease (or condition, etc). 
        * In machine learning terms, this is the proportion of positive examples.
          The expression for prevalence is:
          $$prevalence = \frac{1}{N} \sum_{i} y_i$$
          where $y_i = 1$ when the example is 'positive' (has the disease).
        """
        return np.mean(y_true == 1) # proportion of +ve samples
    
    @staticmethod
    def sensitivity(y_true, y_pred, threshold=0.5):
        """Sensitivity is the probability that our test outputs
            positive given that the case is actually positive."""
        y_pred = y_pred >= threshold
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TP / (TP + FN)

    @staticmethod
    def specificity(y_true, y_pred, threshold=0.5):
        """Specificity is the probability that the test outputs
            negative given that the case is actually negative."""
        y_pred = y_pred >= threshold
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        return TN / (TN + FP)

    @staticmethod
    def ppv(y_true, y_pred, threshold=0.5):
        """Positive predictive value (PPV) is the probability that
            subjects with a positive screening test truly have the disease."""
        y_pred = y_pred >= threshold
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        return TP / (TP + FP)

    @staticmethod
    def npv(y_true, y_pred, threshold=0.5):
        """Negative predictive value (NPV) is the probability that 
            subjects with a negative screening test truly don't have the disease."""
        y_pred = y_pred >= threshold
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TN / (TN + FN)
    

if __name__ == "__main__":
    metrics = Metrics(['tp_tn_fp_fn', 'prevalence', 'sensitivity','specificity', 'acc', 'ppv', 'npv'])
    y_true = np.array([1,1,0,0,0,0,0,0,0,1,1,1,1,1])
    # y_pred = np.array([1,1,0,0,0,0,0,0,0,1,1,1,0,0])
    y_pred = np.array([0.8,0.7,0.4,0.3,0.2,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0])

    print(metrics(y_true, y_pred=y_pred, y_probs=y_pred))
