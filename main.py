from keras.utils import np_utils 
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import numpy as np
import os 
import cv2 as cv
import matplotlib.pyplot as plt
import json

import settings as s

with open('./bread.json', mode='r', encoding='utf8') as jFile:
  bread_types = json.load(jFile)

def getImgsPath(filePath, subfile):
  imgs_list = []
  imgs_path = os.path.join(os.getcwd(), filePath, subfile).replace('\\','/') # 修改windows雙反斜線的路徑
  for fileName in os.listdir(imgs_path):
    img_path = os.path.join(imgs_path, fileName).replace('\\','/') # 修改windows雙反斜線的路徑
    imgs_list.append(img_path)

  return imgs_list

# def showImage(orginImgs, newImgs):
#   plt.figure(num="PCA decomposition")
#   for i in range(len(newImgs) + len(orginImgs)):
#     if i < len(orginImgs): # 一半前是原圖
#       plt.subplot(8, 16, i + 1)
#       plt.imshow(orginImgs[i])
#       plt.axis("off")
#     else: # 一半之後是壓縮圖
#       print(i, "")
#       plt.subplot(8, 16, i + 1)
#       plt.imshow(newImgs[i - len(orginImgs)])
#       plt.axis("off")
    
#   plt.show()

def doPCA(img, num_components):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=num_components)
  transform_imgs = pca.fit_transform(img)
  reconstructed_images = pca.inverse_transform(transform_imgs)
  reconstructed_images = reconstructed_images.astype(np.uint8) # 因為matplot是顯示整數, 需要從float to int

  # print(pca.explained_variance_)
  # print(pca.explained_variance_ratio_) # 顯示主成分的比例

  return reconstructed_images

# 將圖片降維
def preprocessImg(imgs):
  # 將圖片像素flatten
  imgs = np.array(imgs) # 將list轉成numpy array
  flatten_imgs = np.reshape(imgs, (imgs.shape[0], -1))
  
  decompImg = doPCA(flatten_imgs, 64)
  shape = (-1, 128, 128, 3) # 第一個參數是整體圖片的數量
  reshapeImg = np.reshape(decompImg, shape)

  return reshapeImg

def readRGBImg(img_path):
  img = []
  try:
    img_bgr = cv.imread(img_path, cv.IMREAD_COLOR) # 預設讀彩色是BGR圖
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # 轉BGR to RGB
    # origin_imgs.append(img_rgb)
    # showImage(origin_imgs, imgs)
  except Exception as e:
    print(img_path,e)
  
  return img


# 載入麵包的資料 約5000張訓練 1000張預測 (128*128) RGB圖
def loadData():
  # 自動生成訓練集的圖片 每次生成128張
  train_gen = ImageDataGenerator( # 資料增強增加學習樣本
                rotation_range=40, # 角度值，0~180，影象旋轉
                width_shift_range=0.2, # 水平平移，相對總寬度的比例
                height_shift_range=0.2, # 垂直平移，相對總高度的比例
                shear_range=0.2, # 隨機錯切換角度
                zoom_range=0.2, # 隨機縮放範圍
                horizontal_flip=True, # 一半影象水平翻轉
                fill_mode='nearest' # 填充新建立畫素的方法
              ).flow_from_directory(s.TRAINING_FILE_PATH, target_size=(128, 128), batch_size=128)
  # 將資料集少的類別提高權重

  class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_gen.classes), 
            train_gen.classes)
  # 自動生成測試集的圖片 每次生成64張
  test_gen = ImageDataGenerator().flow_from_directory(s.VALIDATION_FILE_PATH, target_size=(128, 128), batch_size=64)

  return train_gen, test_gen, class_weights

  # # 獲得訓練集
  # x_train = []
  # y_train = []
  # # 麵包種類
  # for i in range(len(bread_types)):
  #   training_imgs_path = getImgsPath(s.TRAINING_FILE_PATH, bread_types[i]['type'])
  #   # 某個麵包種類的所有圖片
  #   for img_path in training_imgs_path:
  #     tarining_img = readRGBImg(img_path)
  #     tarining_img = np.array(tarining_img) # tansfer list to numpy array
  #     x_train_normalize = tarining_img / 255  # normalize 0~1
  #     # 儲存圖片資料
  #     x_train.append(x_train_normalize)
  #     # 儲存圖片標籤
  #     y_train.append(i)
  
  # # x_train = preprocessImg(x_train)
  # # tansfer list to numpy array
  # x_train = np.array(x_train)
  # y_trainOneHot = np_utils.to_categorical(y_train, num_classes=s.NUM_CLASSES)

  # # 獲得資料測試集
  # x_test = []
  # y_test = []
  # # 麵包種類
  # for i in range(len(bread_types)):
  #   testing_imgs_path = getImgsPath(s.VALIDATION_FILE_PATH, bread_types[i]['type'])
  #   # 某個麵包種類的所有圖片
  #   for img_path in testing_imgs_path:
  #     testing_img = readRGBImg(img_path)
  #     testing_img = np.array(testing_img) # tansfer list to numpy array
  #     x_testing_normalize = testing_img / 255  # normalize 0~1
  #     # 儲存圖片資料
  #     x_test.append(x_testing_normalize)
  #     # 儲存圖片標籤
  #     y_test.append(i)

  # # x_test = preprocessImg(x_test)
  # # tansfer list to numpy array
  # x_test = np.array(x_test)
  # y_testOneHot = np_utils.to_categorical(y_test, num_classes=s.NUM_CLASSES)

  # return (x_train, y_trainOneHot), (x_test, y_testOneHot)

# 設定CNN model
def loadModel():
  # include_top=False,表示會載入 VGG16 的模型, 全連結層需要自己接
  model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

  # 凍結預設的參數
  # 只需要訓練最後一層10種麵包
  for layer in model.layers:
    layer.trainable = False

  # 打平所有的卷積層
  newModel = Flatten(name='flatten_1')(model.output)
  # 第一層full connected layer
  newModel = Dense(2048, activation='relu', name='block6_dense1')(newModel)
  # 第二層full connected layer
  newModel = Dense(2048, activation='relu', name='block6_dense2')(newModel)
  # 刪除隨機一半的神經元避免彼此合作
  newModel = Dropout(0.5)(newModel)
  # output layer
  newModel = Dense(s.NUM_CLASSES, activation='softmax', name='prediction')(newModel)
  # 重新建立模型結構
  model = Model(model.input, newModel, name='bread_VGG16')

  return model

# 畫訓練成果圖
def show_train_history(train_history, train_acc, test_acc):
  plt.plot(train_history.history[train_acc], label = "train")
  plt.plot(train_history.history[test_acc], label = "test")
  plt.title('Train History')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(loc='upper right')
  plt.show()

if __name__ == "__main__":
  # 載入訓練資料
  # (x_train, y_train_ohe), (x_test, y_test_ohe) = loadData()
  # print(x_train.shape,y_train_ohe.shape)
  # print(x_test.shape, y_test_ohe.shape)
  
  train_gen, test_gen, class_weight = loadData()
  print(train_gen)
  print(dict(enumerate(class_weight)))
  # 載入CNN model
  model = loadModel()
  print(model.summary())

  # 使用adam最佳化
  opt = Adam(lr=0.001)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy']
  )

  # 每次訓練後儲存模型
  checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  # 當訓練集上的loss不在減小的時候停止繼續訓練，因為繼續訓練會導致測試集上的準確率下降。
  earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.0, patience=20, verbose=1, mode='auto')
  
  # 訓練資料
  # train_history=model.fit(x=x_train,y=y_train_ohe,
  #                         batch_size=128, epochs=50, verbose=2,
  #                         validation_data=(x_test, y_test_ohe),
  #                         callbacks=[checkpoint, earlyStop]
  # )
  # 訓練量 steps_per_epoch*epochs*batch_size
  train_history=model.fit_generator(generator=train_gen, 
                        steps_per_epoch=train_gen.n/train_gen.batch_size,
                        epochs=50,
                        class_weight=class_weight,
                        validation_data=test_gen,
                        validation_steps=50,
                        verbose=2,
                        workers=4, # 用多線程載入批次圖片計算
                        callbacks=[checkpoint, earlyStop])

  # 顯示準確率
  show_train_history(train_history, 'accuracy', 'val_accuracy')
  # 顯示失準率
  show_train_history(train_history, 'loss', 'val_loss')
  # 評估模型準確率
  # print('Test accuracy:', model.evaluate(x_test, y_test_ohe)[1])
  print('Test accuracy:', model.evaluate_generator(generator=test_gen, steps=50, verbose = 1)[1])
  