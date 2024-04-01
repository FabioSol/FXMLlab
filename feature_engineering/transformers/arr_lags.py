from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ts_lags(BaseEstimator,TransformerMixin):
    def __init__(self, lag=1):
        self.lag = lag

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy().dropna()
        cols = X.columns
        shifted_cols = [X[c].shift(i) for i in range(1, self.lag + 1) for c in cols]
        X = pd.concat(shifted_cols, axis=1)
        X = X.dropna().values
        return X.reshape(len(X), self.lag, len(cols))

