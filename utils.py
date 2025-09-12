import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def calculate_vif(X, feature_cols):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the design matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix (including intercept as first column).
    feature_cols : list
        Names of features (excluding intercept).

    Returns
    -------
    vif : pd.Series
        VIF values indexed by feature names.
    """
    vif_data = {}
    X = np.asarray(X)
    n_features = X.shape[1] - 1  # Exclude intercept
    
    for j in range(1, n_features + 1):  # Skip intercept (first column)
        # Features excluding the j-th feature
        mask = np.ones(n_features + 1, dtype=bool)
        mask[j] = False
        X_others = X[:, mask]
        X_j = X[:, j]
        
        # Regress feature j against other features
        model = LinearRegression()
        model.fit(X_others, X_j)
        yhat_j = model.predict(X_others)
        
        # Compute R² and VIF
        r2 = r2_score(X_j, yhat_j)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        
        vif_data[feature_cols[j - 1]] = vif
    
    return pd.Series(vif_data)