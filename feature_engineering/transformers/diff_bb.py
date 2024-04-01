from sklearn.base import BaseEstimator, TransformerMixin
from feature_engineering.transformers.bollinger_bands import BBTransformer


class DiffBBTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, col='Close', w1:int=20,w2:int=40,sd1:float=2.5,sd2:float=2.5):
        self.col = col
        self.bb1 = BBTransformer(w1, sd1, self.col)
        self.bb2 = BBTransformer(w2, sd2, self.col)

    def fit(self,X,y=None):
        self.bb1.fit(X, y)
        self.bb2.fit(X, y)
        return self

    def transform(self,X):
        bb1_features = self.bb1.transform(X)
        bb2_features = self.bb2.transform(X)
        X_new=X.copy()[[self.col]].diff()
        for b1,b2 in zip(bb1_features.columns,bb2_features.columns):
            if b1 !=self.col:
                X_new[b1.split("_")[0]+"_diff"]=bb1_features[b1]/bb2_features[b2]-1
        return X_new
