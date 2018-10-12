from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# 训练集与校验集图片大小
img_width, img_height = 28, 28
# 批次大小
train_batch_size = 16
# 轮次（训练次数）
epochs = 50
# 每个分类训练样本数
train_sample = 800
# 每个分类校验样本数
val_sample = 200

# 图片集路径
train_data_dir = 'data/train'
val_data_dir = 'data/validation'

# 使用tensorflow，图片通道在后（channels_last）
input_shape = (img_width, img_height, 3)

# 创建序贯模型
model = Sequential()

# 卷积层
model.add(Convolution2D(
    32,  # 卷积核数
    (3, 3),  # 核的长宽
    input_shape=input_shape  # 输入shape
))

# 激活层
model.add(Activation('relu'))

# 池化
model.add(MaxPooling2D(
    pool_size=(2, 2)  # 下采样尺寸
))

# 第二层
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第三层
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 准备全连接（Full Connection）
model.add(Flatten())  # 数据一维化
model.add(Dense(128))  # 全连接层
model.add(Activation('relu'))  # 激活
model.add(Dropout(0.2))  # 0.2概率断开神经元连接，防止过拟合
model.add(Dense(10))  # 全连接层
model.add(Activation('sigmoid'))

# 编译
model.compile(
    loss='categorical_crossentropy',  # 损失函数
    optimizer='rmsprop',  # 优化器
    metrics=['accuracy']  # 指标
)

# 构建训练图片生成器
train_data_gan = ImageDataGenerator(
    shear_range=0.2,  # 剪切强度（逆时针方向的剪切变换角度）
    zoom_range=0.2  # 随机缩放的幅度
)

train_generator = train_data_gan.flow_from_directory(
    train_data_dir,  # 目录
    target_size=(img_width, img_height),  # 尺寸
    batch_size=train_batch_size,  # 批次大小
    class_mode='categorical'
)

# 构建校验图片生成器
test_data_gen = ImageDataGenerator()

test_generator = test_data_gen.flow_from_directory(
    val_data_dir,  # 目录
    target_size=(img_width, img_height),  # 尺寸
    batch_size=train_batch_size,  # 批次大小
    class_mode='categorical'
)

if __name__ == '__main__':
    # 训练的val_loss在5轮内没有减少，则停止训练
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_sample // train_batch_size,
        epochs=epochs,
        # callbacks=[early_stopping],
        validation_data=test_generator,
        validation_steps=train_sample // train_batch_size
    )
    model.save('First_try.h5')
