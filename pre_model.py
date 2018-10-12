from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

model = load_model('First_try.h5')


def predict(model, img):
    # if img.size != target_size:
    #    img = img.resize(target_size)
    # x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    # x = preprocess_input(x)
    preds = model.predict(x, batch_size=1, verbose=0)
    return preds


def plot_preds(image, preds):
    '''Displays image and the top-n predicted probabilities in a bar graph
    Args:
      image: PIL image
      preds: list of predicted labels and their probabilities
    '''
    plt.imshow(image)
    plt.axis('off')

    plt.figure()
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ylabel('pre')
    plt.bar(classes, preds, color='g')
    plt.xticks(classes)
    plt.show()


if __name__ == '__main__':
    # img = Image.open('data/test/1.png')
    # 装载图片
    sourceImg = cv2.imread('data/test/test.png')
    # 因为训练集的图片是黑底白字，此处时行白黑反转
    ret,changeColor = cv2.threshold(sourceImg,0,255,cv2.THRESH_BINARY_INV)
    # 训练集图片尺寸为28x28，此处修改图片尺寸
    img28x28 = cv2.resize(changeColor, (28, 28))
    # 放进模型中进行识别
    preds = predict(model, img28x28)
    print(preds)
    # 打印结果
    plot_preds(sourceImg, preds[0])

    # print(img.shape)
