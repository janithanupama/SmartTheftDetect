from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM,Dense,Conv2D
from tensorflow.keras.layers import Dropout,MaxPooling2D,Flatten

def create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH,1)))

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(32))

    model.add(Dense(2, activation = 'softmax'))
    model.summary()
    return model

# #%%
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# from keras.layers import Dense, Input
# from keras.layers.pooling import GlobalAveragePooling2D
# from keras.layers import LSTM
# from keras.layers import TimeDistributed

# def create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH):
#     n_classes = 2

#     video = Input(shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH, 3))
#     cnn_base = VGG16(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3),weights="imagenet",include_top=False)

#     cnn_out = GlobalAveragePooling2D()(cnn_base.output)
#     cnn = Model(inputs=cnn_base.input, outputs=cnn_out)
#     cnn.trainable = False
#     encoded_frames = TimeDistributed(cnn)(video)
#     encoded_sequence = LSTM(128)(encoded_frames)
#     hidden_layer = Dense(64, activation="relu")(encoded_sequence)
#     outputs = Dense(n_classes, activation="softmax")(hidden_layer)
#     model = Model([video], outputs)

#     model.summary()
#     return model
