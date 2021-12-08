from keras.layers import Conv2D, Dense, Flatten, Input, add
from keras.layers.core import Activation
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model
from encoder import Encoder
from tiaocan import bot_name


encoder = Encoder()
board_input = Input(shape = encoder.shape(), name = 'board_input')

pb = board_input
pb = Conv2D(16, (3, 3), padding = 'same', data_format = 'channels_first')(pb)
pb = BatchNormalization(axis=1)(pb)
pb = Activation('relu')(pb)

for i in range(4):
    re_pb = pb
    pb = Conv2D(16, (3, 3), padding = 'same', data_format = 'channels_first')(pb)
    pb = BatchNormalization(axis=1)(pb)
    pb = Activation('relu')(pb)

    pb = Conv2D(16, (3, 3), padding = 'same', data_format = 'channels_first')(pb)
    pb = BatchNormalization(axis=1)(pb)
    pb = add([pb, re_pb])
    pb = Activation('relu')(pb)

policy_conv = Conv2D(2, (1, 1), data_format = 'channels_first')(pb)
policy_conv = BatchNormalization(axis=1)(policy_conv)
policy_conv = Activation('relu')(policy_conv)
policy_flat = Flatten()(policy_conv)
policy_output = Dense(encoder.num_moves(), activation = 'softmax')(policy_flat)

value_conv = Conv2D(1, (1, 1), data_format = 'channels_first')(pb)
value_conv = BatchNormalization(axis=1)(value_conv)
value_conv = Activation('relu')(value_conv)
value_flat = Flatten()(value_conv)
value_hidden = Dense(128, activation = 'relu')(value_flat)
value_output = Dense(1, activation = 'tanh')(value_hidden)

model = Model(
    inputs = [board_input],
    outputs = [policy_output, value_output]
)
model.save(bot_name)