import numpy as np
from ROCAnalysis import ROCAnalysis


class ForwardSelection:
    """
    Forward feature selection that greedily adds the feature giving the highest F-score.
    Uses a single 80/20 internal train/test split per round so all candidate features
    in that round are evaluated on the same held-out data.

    Attributes:
        X (ndarray): Full feature matrix.
        y (ndarray): Target labels.
        model (object): ML model with fit() and predict() methods.
        selected_features (list): Feature indices chosen by forward selection.
        best_cost (float): Best F-score achieved during selection.
    """

    def __init__(self, X, y, model):
        """
        Initialize ForwardSelection.

        Parameters:
            X (array-like): Feature matrix (n x p).
            y (array-like): Binary target labels (n,).
            model (object): Classifier with fit(X, y) and predict(X) methods.
        """
        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)
        self.model = model
        self.selected_features = []
        self.best_cost = 0.0

    def create_split(self, X, y):
        """
        Create a random 80/20 train/test split.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Labels.

        Returns:
            X_train, X_test, y_train, y_test (ndarrays)
        """
        n = len(y)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

    def train_model_with_features(self, features, X_train, X_test, y_train, y_test):
        """
        Train the model on a pre-split train set and evaluate F-score on the test set,
        using only the specified feature columns.

        Parameters:
            features (list): Column indices to use.
            X_train, X_test (ndarray): Pre-created train/test feature matrices (full).
            y_train, y_test (ndarray): Corresponding labels.

        Returns:
            float: F1-score on the test split.
        """
        self.model.fit(X_train[:, features], y_train)
        probs = self.model.predict(X_test[:, features])
        y_pred = (probs >= 0.5).astype(int)
        roc = ROCAnalysis(y_pred, y_test)
        return roc.f_score()

    def forward_selection(self):
        """
        Greedily select features one at a time. In each round a single 80/20 split is
        created so all candidates are evaluated on the same held-out set — giving a fair
        comparison. Stops when no remaining feature improves the F-score.
        """
        n_features = self.X.shape[1]
        remaining = list(range(n_features))
        self.selected_features = []
        self.best_cost = 0.0

        while remaining:
            # One split per round — reused for every candidate in this round
            X_train, X_test, y_train, y_test = self.create_split(self.X, self.y)
            best_f = -1.0
            best_feat = None
            for feat in remaining:
                candidate = self.selected_features + [feat]
                f = self.train_model_with_features(candidate, X_train, X_test, y_train, y_test)
                if f > best_f:
                    best_f = f
                    best_feat = feat
            if best_f > self.best_cost:
                self.best_cost = best_f
                self.selected_features.append(best_feat)
                remaining.remove(best_feat)
                print(f"  Added feature {best_feat} -> F-score = {best_f:.4f} "
                      f"(features so far: {self.selected_features})")
            else:
                break

    def fit(self):
        """
        Fit the model on the full sub-dataset using only the selected features.
        """
        X_sel = self.X[:, self.selected_features]
        self.model.fit(X_sel, self.y)

    def predict(self, X_test):
        """
        Predict using the selected features.

        Parameters:
            X_test (array-like): Full feature matrix for test instances.

        Returns:
            array-like: Predicted probabilities.
        """
        X_test = np.array(X_test, dtype=float)
        return self.model.predict(X_test[:, self.selected_features])
