from sklearn.pipeline import Pipeline
from feature_engineering.transformers.arr_lags import ts_lags
from feature_engineering.transformers.Momentum import MomentumTransformer
from  feature_engineering.transformers.y_price import YPriceTransformer
from feature_engineering.transformers.bollinger_bands import BBTransformer

ap_pipeline_x = Pipeline(steps=[
                        ("BB1",BBTransformer(50,2.0)),
                        ("BB2",BBTransformer(15,1.5))])
ap_pipeline_x_expanded = Pipeline(steps=[
                        ("M",MomentumTransformer()),
                        ("BB1",BBTransformer(50,2.0)),
                        ("BB2",BBTransformer(15,1.5))])

ap_pipeline_y = Pipeline(steps=
                       [("y_price",YPriceTransformer())])
def alpha_plus_preprocessing(df,split=0.9):
    X=ap_pipeline_x.fit_transform(df)[:-1]
    y=ap_pipeline_y.fit_transform(df)[len(df)-len(X)-1:]
    pivot=int(len(X)*split)
    return X[:pivot],y[:pivot],X[pivot:],y[pivot:]

