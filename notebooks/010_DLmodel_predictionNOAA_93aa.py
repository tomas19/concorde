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

# %%
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# %%
#### Change inputs
modelID = 93
## noaa stations following same order of the outputs
NOAAstations = ['Duck', 'Oregon', 'Hatteras', 'Beaufort', 'Wilmington', 'Wrightsville']
## station to predict
st = 'Beaufort'

#### Define paths and load data
pathData = Path(r'../data/random_split')
#pathData = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/NNmodel/inputs/random_split')
pathColSample = pathData.parent
#pathColSample = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/adcirc/concorde/batch02/_postprocessing/_preprocessForNN')
X_train_file = 'X_train_standardScaled_allInputs_augmentedAllX10.npy'
Y_train_file = 'y_train_augmentedAllX10.npy'
X_test_file = 'X_test_standardScaled_allInputs_augmentedAllX10.npy'
Y_test_file = 'y_test_augmentedAllX10.npy'

#### some hyperparameters
batch_size = 20
epochs = 250
fold = 1 ## no cross validation

#### path to store outputs
#pathOut0 = Path(r'/mnt/drive1/Insyncs/NCSU/thesis/models/NNmodel/81')
pathOut0 = Path(r'.')
#pathOut0 = pathOut0/st

#### class to save best model
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, pathout, fold, modelID, st):
        super(CustomCallback, self).__init__()
        self.pathout = pathout
        self.fold = fold
        self.modelID = modelID
        self.previous_val_loss = float('inf')  # Initialize with a high value
        self.best_epoch = None
        self.best_model = None
        self.station = st
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None and current_val_loss < self.previous_val_loss:
            self.model.save(self.pathout / f'bestModel_{self.modelID:02d}_fold{self.fold}_{self.station}.tf')
            self.previous_val_loss = current_val_loss
            self.best_epoch = epoch
            self.best_model = self.model
            with open(self.pathout / f'best_model_{self.station}.txt', 'a') as fout:
                fout.write(f'Best model saved for fold {self.fold}: epoch {self.best_epoch}, val_loss: {current_val_loss:0.3f}\n')

#### load data
X_train = np.load(pathData/X_train_file)
y_train = np.load(pathData/Y_train_file)
X_test = np.load(pathData/X_test_file)
y_test = np.load(pathData/Y_test_file)

#### train/validation split
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#### pathout
pathOut = pathOut0/st

columns_sample = pd.read_csv(pathColSample/'dct_tracksAll_batch02_lengthCorr_tides_resampled_SAMPLE.csv', index_col = 0)

## inputs
cols = ['wind_speed', 'pressure', 'rad_to_max_ws', 'forward_speed_u', 'forward_speed_v',
            f'dist_to_{st.lower()}', f'{st}', 'wind_speed_fft', 'pressure_fft', 'rad_to_max_ws_fft',
            'forward_speed_u_fft', 'forward_speed_v_fft']

## extract inputs idx from the full input array
idx_cols = [list(columns_sample).index(x) for x in cols]
X_train_sub2 = X_train_sub[:, :, idx_cols]
X_val2 = X_val[:, :, idx_cols]
X_test2 = X_test[:, :, idx_cols]


## select corresponding output
ixSt = NOAAstations.index(st)
print('\n')
print('************************')
print(f'       {ixSt}          ')
print('************************')

#### select correct y variables
y_train_sub2 = y_train_sub[:, ixSt, 0]
y_val2 = y_val[:, ixSt, 0]
y_test2 = y_test[:, ixSt, 0]

#### make output directory
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
                Dense(1, activation='relu'),
            ])

optimizer = RMSprop(learning_rate = 1e-4)
model.compile(optimizer=optimizer, loss='mse', 
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.summary()

#### callback to save model with lower val_loss
callback = CustomCallback(pathOut, fold=fold, modelID=modelID, st=st)
#### train the model
history = model.fit(X_train_sub2, y_train_sub2, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data = (X_val2, y_val2), callbacks=[callback])


#### save outputs
dfhist = pd.DataFrame.from_dict(history.history)
dfhist.to_csv(pathOut/f'history_{modelID:02d}_{st}_fold{fold}.csv')

#### train validation curve
fig, ax = plt.subplots(figsize = (12, 4))
dfhist[['loss', 'val_loss']].plot(ax = ax)
ax.set_title(f'{st}: fold {fold}')
ax.set_xlabel('Epochs [-]')
ax.set_ylabel('MSE [m2]')
fig.savefig(pathOut/f'trainValCurve_{modelID:02d}_{st}_fold{fold}.png', dpi = 100, 
            bbox_inches = 'tight')

####  predictions
predictions = callback.best_model.predict(X_val2).reshape(-1)
dftest = pd.DataFrame({'y_val_true': y_val2.reshape(-1),
                        f'y_pred_fold{fold}': predictions})

#### scatter plot
fig, ax = plt.subplots(figsize = (4,4))
sns.regplot(dftest, x = 'y_val_true', y = f'y_pred_fold{fold}', ax = ax, fit_reg = False)
ax.plot(np.arange(0, 5, 0.5), np.arange(0, 5, 0.5), ls = '--', c = 'k')
ax.set_title(f'{st}: fold {fold}')
fig.savefig(pathOut/f'predValSet_{modelID:02d}_{st}_fold{fold}.png', 
            dpi = 100, bbox_inches = 'tight')

#### delete model
tf.keras.backend.clear_session()
del model
del optimizer
del callback
