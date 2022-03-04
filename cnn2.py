
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, Dropout, LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras.callbacks import TensorBoard
import numpy as np
import preprocess
import plotcm

# 训练参数
batch_size = 128
epochs = 20
num_classes = 16
length = 2048
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.5,0.25,0.25] # 测试集验证集划分比例


path = r'data'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28)

x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]

input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

model_name = "cnn2"

# 实例化一个Sequential
model = Sequential()
#第一层卷积
model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
#第二层卷积
model.add(Conv1D(32,kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
#LSTM层
model.add(LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True))
# 展平
model.add(Flatten())
model.add(Dropout(0.2))
# 添加全连接层
model.add(Dense(32))
model.add(Activation("relu"))
# 增加输出层
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
model.summary()

# 编译模型
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard调用
tb_cb = TensorBoard(log_dir='logs/{}'.format(model_name))

# 开始模型训练
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
          callbacks=[tb_cb])

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("loss：", score[0])
print("accuracy：", score[1])
plot_model(model=model, to_file='images/cnn2.png', show_shapes=True)

labels = ['118', '185', '222', '3005', '105', '169', '209', '3001',
          '144',
          '130',
          '156', '197', '246', '234', '258', '97',
          ]
plotcm.plot_confuse(model, x_test, y_test, labels)