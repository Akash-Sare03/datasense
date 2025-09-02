import pandas as pd
import pytest
from datasense.feature_importance import feature_importance_calculate

class TestFeatureImportance:
    """Test suite for feature_importance.py"""
    
    def test_feature_importance_classification(self):
        """Test with classification data"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Binary classification
        })
        
        result = feature_importance_calculate(df, 'target', top_n=2)
        # Should return target_type = "classification"
        assert result == "classification"
    
    def test_feature_importance_regression(self):
        """Test with regression data"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'target': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Regression
        })
        
        result = feature_importance_calculate(df, 'target', top_n=2)
        # Should return target_type = "regression"
        assert result == "regression"
    
    def test_feature_importance_missing_target(self):
        """Test with missing target column"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            feature_importance_calculate(df, 'nonexistent')