from abc import ABC, abstractmethod


class RiskEstimatorBase(ABC):
    """
    Abstract base class for all credit default prediction models.

    This class defines the interface that all models must implement.
    It ensures that every model can be trained, can predict probabilities,
    and can generate a consistent metrics report.
    """

    def __init__(self, random_seed=2026):
        """
        Initialize the base estimator with a fixed random seed.

        Args:
            random_seed: Integer seed for reproducible results
        """
        self.random_seed = random_seed
        self.fitted_engine = None # will hold actual model
        self.training_features = None # will store feature names consistency

    @abstractmethod
    def train(self, feature_matrix, target_vector):
        """
        Train (fit) the model on the provided data.

        Every subclass MUST implement this method.

        Args:
            feature_matrix: pandas DataFrame or numpy array of predictor variables
            target_vector: pandas Series or numpy array of binary target (0=no default, 1=default)

        Returns:
            self (the trained model instance)
        """
        pass

    @abstractmethod
    def forecast_default_proba(self, feature_matrix):
        """
        Predict the probability of default (class 1) for each sample.

        Every subclass MUST implement this method.

        Args:
            feature_matrix: pandas DataFrame or numpy array of predictor variables

        Returns:
            numpy array of probabilities (values between 0 and 1)
        """
        pass

    @abstractmethod
    def forecast_default_class(self, feature_matrix, threshold=0.5):
        """
        Predict binary class labels (0 or 1) based on a probability threshold.

        Every subclass MUST implement this method.

        Args:
            feature_matrix: pandas DataFrame or numpy array of predictor variables
            threshold: float between 0 and 1 (default 0.5)

        Returns:
            numpy array of 0/1 predictions
        """
        pass

    @abstractmethod
    def compute_all_metrics(self, feature_matrix, true_targets):
        """
        Calculate all six required evaluation metrics.

        Every subclass MUST implement this method.

        The metrics returned are:
            1. accuracy_ratio      -> Accuracy
            2. auc_roc_score       -> AUC Score
            3. precision_ratio     -> Precision
            4. recall_ratio        -> Recall
            5. f1_measure          -> F1 Score
            6. matthews_correlation -> MCC Score

        Args:
            feature_matrix: pandas DataFrame or numpy array of predictor variables
            true_targets: pandas Series or numpy array of true labels

        Returns:
            dict: Metric names as keys, rounded values as values
        """
        pass
