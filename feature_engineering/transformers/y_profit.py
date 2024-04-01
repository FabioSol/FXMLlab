from sklearn.base import BaseEstimator, TransformerMixin



class YPriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col='Close', sl=.01,tp=.01,max_p=10):
        self.sl=sl
        self.tp=tp
        self.max_p=max_p
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()[[self.col]]


        return X

