from abc import ABC, abstractmethod
import numpy as np


class MachineLearningModel(ABC):
    """Abstract base class for machine learning models."""

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass


def _polynomial_features(self, X):
    """
    Generate polynomial features from the input features.
    Module-level helper — 'self' is unused here; use _build_poly_features instead.
    """
    pass


def _build_poly_features(X, degree):
    """
    Build [1, X, X^2, ..., X^degree] for multivariate X.
    Each feature column is independently raised to powers 1..degree.

    Parameters:
    X (ndarray): shape (n, p) feature matrix (no bias column).
    degree (int): polynomial degree.

    Returns:
    Xe (ndarray): shape (n, 1 + p*degree) extended feature matrix.
    """
    n = X.shape[0]
    Xe = np.ones((n, 1))
    for d in range(1, degree + 1):
        Xe = np.hstack([Xe, X ** d])
    return Xe


class RegressionModelNormalEquation(MachineLearningModel):
    """Regression using the Normal Equation for polynomial regression of any degree."""

    def __init__(self, degree):
        """
        Initialize the model.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        self.degree = degree
        self.beta = None
        self.cost = None

    def fit(self, X, y):
        """
        Train using the Normal Equation: beta = (X^T X)^{-1} X^T y.
        Stores beta and the training MSE in self.cost.

        Parameters:
        X (array-like): Features (n x p).
        y (array-like): Target values (n,).
        """
        X = np.atleast_2d(X) if np.ndim(X) == 1 else np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y, dtype=float)
        Xe = _build_poly_features(X, self.degree)
        self.beta = np.linalg.pinv(Xe.T @ Xe) @ Xe.T @ y
        r = Xe @ self.beta - y
        self.cost = float(r @ r) / len(y)

    def predict(self, X):
        """
        Make predictions by applying the fitted polynomial model.

        Parameters:
        X (array-like): Features (m x p).

        Returns:
        predictions (ndarray): Predicted values (m,).
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xe = _build_poly_features(X, self.degree)
        return Xe @ self.beta

    def evaluate(self, X, y):
        """
        Compute MSE on the given data.

        Parameters:
        X (array-like): Features.
        y (array-like): True target values.

        Returns:
        score (float): Mean Squared Error.
        """
        y = np.array(y, dtype=float)
        r = self.predict(X) - y
        return float(r @ r) / len(y)


class RegressionModelGradientDescent(MachineLearningModel):
    """Regression using gradient descent for polynomial features of any degree."""

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Step size for gradient descent.
        num_iterations (int): Number of gradient descent iterations.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train using batch gradient descent.
        Update rule: beta_{j+1} = beta_j - (2*lr/n) * X^T (X*beta - y).
        Tracks cost at each iteration in self.cost_history.

        Parameters:
        X (array-like): Features (n x p).
        y (array-like): Target values (n,).
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y, dtype=float)
        Xe = _build_poly_features(X, self.degree)
        n, p = Xe.shape
        self.beta = np.zeros(p)
        self.cost_history = []
        alpha = self.learning_rate
        for _ in range(self.num_iterations):
            r = Xe @ self.beta - y
            self.cost_history.append(float(r @ r) / n)
            self.beta -= (2.0 * alpha / n) * (Xe.T @ r)

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Parameters:
        X (array-like): Features.

        Returns:
        predictions (ndarray): Predicted values.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xe = _build_poly_features(X, self.degree)
        return Xe @ self.beta

    def evaluate(self, X, y):
        """
        Compute MSE on the given data.

        Parameters:
        X (array-like): Features.
        y (array-like): True target values.

        Returns:
        score (float): Mean Squared Error.
        """
        y = np.array(y, dtype=float)
        r = self.predict(X) - y
        return float(r @ r) / len(y)


class LogisticRegression:
    """Binary Logistic Regression using gradient descent (two classes: 0 or 1)."""

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model.

        Parameters:
        learning_rate (float): Step size for gradient descent.
        num_iterations (int): Number of gradient descent iterations.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid: 1 / (1 + e^{-z}).

        Parameters:
        z (array-like): Input values.

        Returns:
        result (ndarray): Sigmoid output in (0, 1).
        """
        # Clamp to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _cost_function(self, Xe, y):
        """
        Log-loss: J = -(1/n) * [y^T log(h) + (1-y)^T log(1-h)].

        Parameters:
        Xe (ndarray): Extended feature matrix with bias column.
        y (ndarray): Binary labels.

        Returns:
        cost (float): Logistic regression cost.
        """
        eps = 1e-15
        n = len(y)
        h = self._sigmoid(Xe @ self.beta)
        return -(1.0 / n) * (y @ np.log(h + eps) + (1.0 - y) @ np.log(1.0 - h + eps))

    def fit(self, X, y):
        """
        Train using gradient descent.
        Adds bias column internally. Tracks cost in self.cost_history.
        Update rule: beta -= (lr/n) * X^T (h - y).

        Parameters:
        X (array-like): Normalized features (n x p).
        y (array-like): Binary labels (n,).
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        Xe = np.c_[np.ones(len(X)), X]
        n, p = Xe.shape
        self.beta = np.zeros(p)
        self.cost_history = []
        for _ in range(self.num_iterations):
            self.cost_history.append(self._cost_function(Xe, y))
            h = self._sigmoid(Xe @ self.beta)
            self.beta -= (self.learning_rate / n) * (Xe.T @ (h - y))

    def predict(self, X):
        """
        Return predicted probabilities using the sigmoid model.

        Parameters:
        X (array-like): Features (m x p).

        Returns:
        predictions (ndarray): Probabilities in (0, 1).
        """
        X = np.array(X, dtype=float)
        Xe = np.c_[np.ones(len(X)), X]
        return self._sigmoid(Xe @ self.beta)

    def evaluate(self, X, y):
        """
        Compute classification accuracy (threshold at 0.5).

        Parameters:
        X (array-like): Features.
        y (array-like): True binary labels.

        Returns:
        score (float): Fraction of correctly classified samples.
        """
        y = np.array(y)
        y_pred = (self.predict(X) >= 0.5).astype(int)
        return float(np.mean(y_pred == y))


class NonLinearLogisticRegression:
    """
    Non-linear Logistic Regression using polynomial feature mapping for 2 input features.
    Works for binary classification (0 or 1).
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model.

        Parameters:
        degree (int): Degree of polynomial feature mapping.
        learning_rate (float): Step size for gradient descent.
        num_iterations (int): Number of gradient descent iterations.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid.

        Parameters:
        z (array-like): Input values.

        Returns:
        result (ndarray): Sigmoid output in (0, 1).
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def mapFeature(self, X1, X2, D):
        """
        Map two features to a higher-dimensional polynomial space.
        Produces [1, X1, X2, X1^2, X1*X2, X2^2, ...] up to degree D.

        Parameters:
        X1 (array-like): First feature (N,).
        X2 (array-like): Second feature (N,).
        D (int): Polynomial degree.

        Returns:
        Xe (ndarray): Extended feature matrix (N x num_features).
        """
        X1 = np.array(X1).ravel()
        X2 = np.array(X2).ravel()
        Xe = np.c_[np.ones(len(X1)), X1, X2]
        for i in range(2, D + 1):
            for j in range(0, i + 1):
                Xnew = (X1 ** (i - j) * X2 ** j).reshape(-1, 1)
                Xe = np.append(Xe, Xnew, axis=1)
        return Xe

    def _cost_function(self, Xe, y):
        """
        Log-loss cost function.

        Parameters:
        Xe (ndarray): Extended feature matrix.
        y (ndarray): Binary labels.

        Returns:
        cost (float): Logistic regression cost.
        """
        eps = 1e-15
        n = len(y)
        h = self._sigmoid(Xe @ self.beta)
        return -(1.0 / n) * (y @ np.log(h + eps) + (1.0 - y) @ np.log(1.0 - h + eps))

    def fit(self, X, y):
        """
        Train using gradient descent with polynomial feature mapping.
        Tracks cost in self.cost_history.

        Parameters:
        X (array-like): Feature matrix (n x 2).
        y (array-like): Binary labels (n,).
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        X1, X2 = X[:, 0], X[:, 1]
        Xe = self.mapFeature(X1, X2, self.degree)
        n, p = Xe.shape
        self.beta = np.zeros(p)
        self.cost_history = []
        for _ in range(self.num_iterations):
            self.cost_history.append(self._cost_function(Xe, y))
            h = self._sigmoid(Xe @ self.beta)
            self.beta -= (self.learning_rate / n) * (Xe.T @ (h - y))

    def predict(self, X):
        """
        Return predicted probabilities using the polynomial sigmoid model.

        Parameters:
        X (array-like): Feature matrix (m x 2).

        Returns:
        predictions (ndarray): Probabilities in (0, 1).
        """
        X = np.array(X, dtype=float)
        X1, X2 = X[:, 0], X[:, 1]
        Xe = self.mapFeature(X1, X2, self.degree)
        return self._sigmoid(Xe @ self.beta)

    def evaluate(self, X, y):
        """
        Compute classification accuracy (threshold at 0.5).

        Parameters:
        X (array-like): Feature matrix (m x 2).
        y (array-like): True binary labels.

        Returns:
        score (float): Fraction of correctly classified samples.
        """
        y = np.array(y)
        y_pred = (self.predict(X) >= 0.5).astype(int)
        return float(np.mean(y_pred == y))
