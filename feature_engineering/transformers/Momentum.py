from sklearn.base import BaseEstimator, TransformerMixin
from ta.momentum import *


class MomentumTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, aws_w1=5, aws_w2=34, kma_window=10, kma_pow1=2, kma_pow2=20, ppo_wslow=26, ppo_wfast=12,
                 ppo_wsign=9, roc_window=12, rsi_window=14,srsi_window=14,srsi_smooth1=3,srsi_smooth2=4, open='Open', high='High', low='Low', close='Close',
                 col='Close'):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.col = col

        self.aws_w1 = aws_w1
        self.aws_w2 = aws_w2

        self.kma_window = kma_window
        self.kma_pow1 = kma_pow1
        self.kma_pow2 = kma_pow2

        self.ppo_wslow = ppo_wslow
        self.ppo_wfast = ppo_wfast
        self.ppo_wsign = ppo_wsign

        self.roc_window = roc_window

        self.rsi_window = rsi_window

        self.srsi_window=srsi_window
        self.srsi_smooth1=srsi_smooth1
        self.srsi_smooth2=srsi_smooth2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        aws = AwesomeOscillatorIndicator(high=X[self.high], low=X[self.low], window1=self.aws_w1, window2=self.aws_w2)
        X[f'aws_{self.aws_w1}_{self.aws_w2}'] = aws.awesome_oscillator()

        kma = KAMAIndicator(close=X[self.col], window=self.kma_window, pow1=self.kma_pow1, pow2=self.kma_pow2)
        X[f"kma_{self.kma_window}_{self.kma_pow1}_{self.kma_pow2}"] = kma.kama()

        ppo = PercentagePriceOscillator(close=X[self.col], window_slow=self.ppo_wslow, window_fast=self.ppo_wfast,
                                        window_sign=self.ppo_wsign)
        X[f"ppo_{self.ppo_wslow}_{self.ppo_wfast}_{self.ppo_wsign}"] = ppo.ppo()

        roc = ROCIndicator(close=X[self.col], window=self.roc_window)
        X[f"roc_{self.roc_window}"]=roc.roc()

        rsi = RSIIndicator(close=X[self.col], window=self.rsi_window)
        X[f'rsi_{self.rsi_window}'] = rsi.rsi()

        srsi = StochRSIIndicator(close=X[self.close],window=self.srsi_window,smooth1=self.srsi_smooth1,smooth2=self.srsi_smooth2)
        X[f'srsi_{self.srsi_window}_{self.srsi_smooth1}_{self.srsi_smooth2}']=srsi.stochrsi()
        X[f'srsi_{self.srsi_window}_{self.srsi_smooth1}_{self.srsi_smooth2}_d']=srsi.stochrsi_d()
        X[f'srsi_{self.srsi_window}_{self.srsi_smooth1}_{self.srsi_smooth2}_k']=srsi.stochrsi_k()
        return X
