from sklearn.base import BaseEstimator, TransformerMixin
from ta.volatility import BollingerBands


class BBTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, window,window_dev, col='Close'):
        self.col = col
        self.window=window
        self.window_dev=window_dev

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        bb=BollingerBands(close=X[self.col],window=self.window, window_dev=self.window_dev)
        X[f'bbh_{self.window}_{self.window_dev}'] = bb.bollinger_hband()
        X[f'bbl_{self.window}_{self.window_dev}'] = bb.bollinger_lband()
        return X