from sklearn.base import BaseEstimator, TransformerMixin


class YDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col='Close'):
        self.col = col


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()[[self.col]]
        return X.diff().shift(-1).dropna().values.reshape(len(X)-1)
