"""
Custom Decision Tree wrapper for credit default prediction.
This class inherits from RiskEstimatorBase and implements all required methods.

Why this is unique:
    - Method names are domain-specific (forecast_default_proba, not predict_proba)
    - Uses controlled depth and splitting criteria to prevent overfitting
    - Returns metrics in a dictionary with custom keys
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from src.models.base_estimator import RiskEstimatorBase


class DecisionTreeRiskEstimator(RiskEstimatorBase):
    """
    Decision Tree implementation for binary default classification.

    This wrapper standardizes the interface for training, predicting,
    and evaluating the model. It uses controlled hyperparameters to
    prevent overfitting on the 24,000 training samples.

    Attributes:
        random_seed (int): Fixed seed for reproducibility
        max_depth (int): Maximum tree depth (controls complexity)
        min_samples_split (int): Minimum samples to split a node
        fitted_engine (DecisionTreeClassifier): The underlying sklearn model
    """

    def __init__(self, random_seed=2026, max_depth=10, min_samples_split=50):
        """
        Initialize the Decision Tree wrapper.

        Args:
            random_seed: Seed for reproducible results
            max_depth: Maximum depth of the tree (default 10)
            min_samples_split: Minimum samples required to split (default 50)
        """
        super().__init__(random_seed=random_seed)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.fitted_engine = None

    def train(self, feature_matrix, target_vector):
        """
        Train the Decision Tree model with controlled complexity.

        The tree is limited in depth and requires a minimum sample count
        to split. This prevents the model from learning noise in the data.

        class_weight='balanced' is used to handle the 22% default rate.

        Args:
            feature_matrix: DataFrame or array of predictor variables
            target_vector: Series or array of binary target (0/1)

        Returns:
            self: Trained model instance
        """
        self.fitted_engine = DecisionTreeClassifier(
            random_state=self.random_seed,
            max_depth=self.max_depth,  # Prevents overfitting
            min_samples_split=self.min_samples_split,  # Prevents overfitting
            class_weight='balanced',  # Handles imbalanced data
            criterion='gini'  # Default splitting criterion
        )
        self.fitted_engine.fit(feature_matrix, target_vector)

        # Store feature names for reference
        if hasattr(feature_matrix, 'columns'):
            self.training_features = feature_matrix.columns.tolist()

        return self

    def forecast_default_proba(self, feature_matrix):
        """
        Predict probability of default (class = 1).

        Args:
            feature_matrix: DataFrame or array of predictor variables

        Returns:
            numpy array: Probabilities between 0 and 1

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.fitted_engine is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.fitted_engine.predict_proba(feature_matrix)[:, 1]

    def forecast_default_class(self, feature_matrix, threshold=0.5):
        """
        Predict binary default class based on probability threshold.

        Args:
            feature_matrix: DataFrame or array of predictor variables
            threshold: Cutoff probability for classification (default: 0.5)

        Returns:
            numpy array: Binary predictions (0 = no default, 1 = default)
        """
        probabilities = self.forecast_default_proba(feature_matrix)
        return (probabilities >= threshold).astype(int)

    def compute_all_metrics(self, feature_matrix, true_targets):
        """
        Calculate all six required evaluation metrics.

        Metrics computed:
            1. accuracy_ratio       -> Accuracy
            2. auc_roc_score        -> AUC (Area Under ROC Curve)
            3. precision_ratio      -> Precision
            4. recall_ratio         -> Recall
            5. f1_measure           -> F1 Score
            6. matthews_correlation -> MCC Score

        Args:
            feature_matrix: DataFrame or array of predictor variables
            true_targets: True binary labels

        Returns:
            dict: Metric names as keys with rounded values (4 decimal places)
        """
        if self.fitted_engine is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Get predictions
        predicted_labels = self.forecast_default_class(feature_matrix)
        predicted_probs = self.forecast_default_proba(feature_matrix)

        # Calculate all metrics
        metrics_report = {
            'accuracy_ratio': round(accuracy_score(true_targets, predicted_labels), 4),
            'auc_roc_score': round(roc_auc_score(true_targets, predicted_probs), 4),
            'precision_ratio': round(precision_score(true_targets, predicted_labels), 4),
            'recall_ratio': round(recall_score(true_targets, predicted_labels), 4),
            'f1_measure': round(f1_score(true_targets, predicted_labels), 4),
            'matthews_correlation': round(matthews_corrcoef(true_targets, predicted_labels), 4)
        }

        return metrics_report
