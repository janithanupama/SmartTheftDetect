from create_model import create_model
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

#%%
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT,IMAGE_WIDTH = 100,100
seed_constant = 10

#%%
data = np.load('Path/data.npy') # Path to data.npy
target = np.load('Path/label.npy') # Path to label.npy

print(data.shape)
print(target.shape)

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.3,random_state=seed_constant,shuffle=True)

#%%
model = create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH)
plot_model(model, to_file='model.png', show_shapes=True)

early_stopping_callback = EarlyStopping(monitor = 'val_loss',patience = 10, mode = 'min', restore_best_weights = True)
save_best_model_checkpoint = ModelCheckpoint('models/model-{epoch:03d}.h5',monitor='val_loss',save_best_only=True,mode='auto')
bs =4
n_epochs = 100
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
model_training_history = model.fit(x=x_train,y = y_train, epochs = n_epochs, batch_size = bs,shuffle = True, validation_data = (x_test,y_test), callbacks = [early_stopping_callback,save_best_model_checkpoint])

#%%
model.save('models/Final_weights.h5') # Model Weight Save Path

#%%
model_evaluation_history = model.evaluate(x_test,y_test)
model_loss, model_accuracy = model_evaluation_history
print(model_loss)
print(model_accuracy)

#%%
def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    epochs = range(len(metric_value_1))

    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

    plt.title(str(plot_name))
    plt.legend()

    
#%%
plot_metric(model_training_history,'loss','val_loss','Loss Vs Val Loss')
plot_metric(model_training_history,'accuracy','val_accuracy','Accuracy Vs Val Accuracy')





