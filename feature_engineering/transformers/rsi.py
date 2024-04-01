from sklearn.base import BaseEstimator, TransformerMixin
from ta.momentum import RSIIndicator



class RsiTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, window=14, col='Close'):
        self.col = col
        self.window=window

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        rsi=RSIIndicator(close=X[self.col],window=self.window)
        X[f'rsi_{self.window}'] = rsi.rsi()
        return X