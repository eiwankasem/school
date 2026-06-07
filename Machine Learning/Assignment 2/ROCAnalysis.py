class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis and compute confusion matrix counts.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_pred = list(y_predicted)
        self.y_true = list(y_true)
        self.tp = sum(1 for p, t in zip(self.y_pred, self.y_true) if p == 1 and t == 1)
        self.tn = sum(1 for p, t in zip(self.y_pred, self.y_true) if p == 0 and t == 0)
        self.fp = sum(1 for p, t in zip(self.y_pred, self.y_true) if p == 1 and t == 0)
        self.fn = sum(1 for p, t in zip(self.y_pred, self.y_true) if p == 0 and t == 1)

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity / Recall): TP / (TP + FN).

        Returns:
            float: True Positive Rate, or 0.0 if undefined.
        """
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def fp_rate(self):
        """
        Calculate False Positive Rate: FP / (FP + TN).

        Returns:
            float: False Positive Rate, or 0.0 if undefined.
        """
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    def precision(self):
        """
        Calculate Precision: TP / (TP + FP).

        Returns:
            float: Precision, or 0.0 if undefined.
        """
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def f_score(self, beta=1):
        """
        Calculate the F-score: (1 + beta^2) * precision * recall / (beta^2 * precision + recall).

        Parameters:
            beta (float): Weighting factor. beta=1 gives F1 (harmonic mean of precision and recall).

        Returns:
            float: F-score, or 0.0 if undefined.
        """
        p = self.precision()
        r = self.tp_rate()
        denom = beta ** 2 * p + r
        return (1 + beta ** 2) * p * r / denom if denom > 0 else 0.0

    def accuracy(self):
        """
        Calculate overall accuracy: (TP + TN) / total.

        Returns:
            float: Accuracy.
        """
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0
