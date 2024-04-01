from sklearn.base import BaseEstimator, TransformerMixin


class LagTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, lag=1, col='Close'):
        self.col = col
        self.lag = lag

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        X[f'{self.col}_lag_{self.lag}'] = X[self.col].shift(periods=self.lag)
        return X