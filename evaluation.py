import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np
import json

with open('./bread.json', mode='r', encoding='utf8') as jFile:
  bread_types = json.load(jFile)

def decode_predictions(preds, top = 3):
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1] # 將預測結果
    result = [(bread_types[i]['type'] , (pred[i]*100)) for i in top_indices]
    results.append(result)
  
  return results
if __name__ == "__main__":
  # 載入模型
  model = keras.models.load_model('vgg16_1.h5')
  # 顯示模型結構
  # print(model.summary())

  img_path = 'img5.jpg'
  img = image.load_img(img_path, target_size=(128, 128))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  features = model.predict(x)
  # 取得前三個最可能的類別及機率
  predicts = decode_predictions(features, top=3)[0]
  print('Predicted:')
  for (i, (label, prob)) in enumerate(predicts):
    print('{number}. {label} : {prob:.2f}%'.format(number=i + 1, label=label, prob=prob))