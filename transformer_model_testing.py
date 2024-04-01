import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from Loader.ReadCSV import Reader
from feature_engineering.pipelines.AlphaPlusPipeline import alpha_plus_preprocessing
from Models.transformer import transformer_model
from sklearn.preprocessing import StandardScaler



data = Reader.read_file(timeframe="H1").dropna()
X_train,Y_train,X_test,Y_test=alpha_plus_preprocessing(data, 0.7)


input_shape = (X_train.shape[1], X_train.shape[2])
num_layers = 30
dff = 512
d_model = 13
num_heads = 13
dropout = 0.1
output_dim = 1

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


model = transformer_model(input_shape, num_layers, dff, d_model, num_heads, dropout, output_dim)
model.compile(optimizer='adam', loss=rmse)


history = model.fit(x=X_train,y=Y_train,epochs=100,validation_data=(X_test,Y_test),shuffle=False)

axes=plt.axes()
axes.plot(pd.DataFrame(model.history.history)['loss'], label='Loss')
axes.plot(pd.DataFrame(model.history.history)['val_loss'], label='Validation Loss')
axes.legend(loc=0)
axes.set_title('Model fitting performance')
plt.show()

Y_predicted=(model.predict(X_test))
Y_true=(Y_test.reshape(Y_test.shape[0],1))

axes=plt.axes()
axes.plot(Y_true, label='True Y')
axes.plot(Y_predicted, label='Predicted Y')
axes.legend(loc=0)
axes.set_title('Prediction adjustment')
plt.show()

from sklearn import metrics
print('Model accuracy (%)')
Y_p=(model.predict(X_train))
Y_t=(Y_train.reshape(Y_train.shape[0],1))
print((1-(metrics.mean_absolute_error(Y_t, Y_p)/Y_t.mean()))*100)
print('')
print('Prediction performance')
print('MAE in %', (metrics.mean_absolute_error(Y_true, Y_predicted)/Y_true.mean())*100)
print('MSE', metrics.mean_squared_error(Y_true, Y_predicted))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_true, Y_predicted)))
print('R2', metrics.r2_score(Y_true, Y_predicted))


