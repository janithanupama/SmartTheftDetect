#%%
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import LSTM
from keras.layers import TimeDistributed

def create_model(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH):
    n_classes = 2

    video = Input(shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH, 3))
    cnn_base = VGG16(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3),weights="imagenet",include_top=False)

    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(inputs=cnn_base.input, outputs=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(128)(encoded_frames)
    hidden_layer = Dense(64, activation="relu")(encoded_sequence)
    outputs = Dense(n_classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)

    model.summary()
    return model
