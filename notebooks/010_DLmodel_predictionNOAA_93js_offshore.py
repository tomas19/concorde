# %%
# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Masking
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K
import time

# %%
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# %%
#### Change inputs
modelID = '93jx'
## noaa stations following same order of the outputs
NOAAstations = ['Duck', 'Oregon', 'Hatteras', 'Beaufort', 'Wilmington', 'Wrightsville']

#### Define paths and load data
# pathData = Path(r'../../../data/random_split')
pathData = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/NNmodel/inputs/random_split')
# pathColSample = pathData.parent
pathColSample = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/adcirc/concorde/batch02/_postprocessing/_preprocessForNN')
X_train_file = 'X_train_standardScaled_allInputs_augmentedAllX50_def.npy'
Y_train_file = 'y_train_augmentedAllX50_def.npy'
X_test_file = 'X_test_standardScaled_allInputs_augmentedAllX50_def.npy'
Y_test_file = 'y_test_augmentedAllX50_def.npy'

#### some hyperparameters
batch_size = 100
epochs = 1000
fold = 1 ## no cross validation


# %%
#### path to store outputs
#pathOut0 = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/NNmodel/81')
pathOut = Path(f'../models/NNmodel/1DCNN_final_architecture/fftAndLocalTides/{modelID}')
#pathOut0 = pathOut0/st

#### class to save best model
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, pathout, fold, modelID):
        super(CustomCallback, self).__init__()
        self.pathout = pathout
        self.fold = fold
        self.modelID = modelID
        self.previous_val_loss = float('inf')  # Initialize with a high value
        self.best_epoch = None
        self.best_model = None
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None and current_val_loss < self.previous_val_loss:
            self.model.save(self.pathout / f'bestModel_{self.modelID}_fold{self.fold}.tf')
            self.previous_val_loss = current_val_loss
            self.best_epoch = epoch
            self.best_model = self.model
            with open(self.pathout / f'best_model.txt', 'a') as fout:
                fout.write(f'Best model saved for fold {self.fold}: epoch {self.best_epoch}, val_loss: {current_val_loss:0.3f}\n')

#### load data
X_train = np.load(pathData/X_train_file)
y_train = np.load(pathData/Y_train_file)
X_test = np.load(pathData/X_test_file)
y_test = np.load(pathData/Y_test_file)

#### train/validation split
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#### pathout
#pathOut = pathOut0/st

columns_sample = pd.read_csv(pathColSample/'dct_tracksAll_batch02_lengthCorr_tides_resampled_SAMPLE.csv', index_col = 0)

# inputs
cols = ['wind_speed', 'pressure', 'rad_to_max_ws', 'forward_speed_u', 'forward_speed_v',
            'dist_to_duck', 'dist_to_oregon', 'dist_to_hatteras', 'dist_to_beaufort', 'dist_to_wilmington', 'dist_to_wrightsville', 'Boundary', 
            'wind_speed_fft', 'pressure_fft', 'rad_to_max_ws_fft',
            'forward_speed_u_fft', 'forward_speed_v_fft']

# %%
## extract inputs idx from the full input array
idx_cols = [list(columns_sample).index(x) for x in cols]
X_train_sub2 = X_train_sub[:, :, idx_cols]
X_val2 = X_val[:, :, idx_cols]
X_test2 = X_test[:, :, idx_cols]

# %%
try:
    os.mkdir(pathOut)
except:
    pass

#### Define model
model = Sequential([
                Masking(-9999, input_shape=(X_train_sub2.shape[1:])),
                Conv1D(16, kernel_size=3, activation='relu'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Conv1D(32, kernel_size=3, activation='relu'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Conv1D(64, kernel_size=3, activation='relu'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(6, activation='relu'),
            ])

optimizer = RMSprop(learning_rate = 1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), 
                metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.summary()

#### callback to save model with lower val_loss
callback = CustomCallback(pathOut, fold=fold, modelID=modelID)
#### train the model
t0 = time.time()
history = model.fit(X_train_sub2, y_train_sub, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data = (X_val2, y_val), callbacks=[callback])
print(f'Training time: {(time.time() - t0)/3600:0.3f} hrs')

#### save outputs
dfhist = pd.DataFrame.from_dict(history.history)
dfhist.to_csv(pathOut/f'history_{modelID}_fold{fold}.csv')

# %%
fig, ax = plt.subplots(figsize = (12, 4))
dfhist[['loss', 'val_loss']].plot(ax = ax)
ax.set_xlabel('Epochs [-]')
ax.set_ylabel('MSE [m2]')
fig.savefig(pathOut/f'trainValCurve_{modelID}_fold{fold}.png', dpi = 100, 
            bbox_inches = 'tight')

# %%
predictions = callback.best_model.predict(X_val2)

# %%
dfVal = pd.DataFrame(y_val.reshape(y_val.shape[:2]), columns = NOAAstations)
dfPred = pd.DataFrame(predictions, columns = [f'{x}_pred' for x in NOAAstations])
dfAll = pd.concat([dfVal, dfPred], axis = 1)
dfAll.to_csv(pathOut/f'predValSet_{modelID}.csv')

# %%
for i in range(6):
    fig, ax = plt.subplots(figsize = (4,4))
    sns.regplot(x = dfVal.iloc[:, i], y = dfPred.iloc[:, i], ax = ax, fit_reg = False)
    ax.plot(np.arange(0, 5, 0.5), np.arange(0, 5, 0.5), ls = '--', c = 'k')
    ax.set_title(f'{NOAAstations[i]}')
    fig.savefig(pathOut/f'predValSet_{modelID}_{NOAAstations[i]}.png',
                dpi = 100, bbox_inches = 'tight')

# %%
#### delete model
tf.keras.backend.clear_session()
del model
del optimizer
del callback

# %%



