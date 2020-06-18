# Bread Image Analyzation
> 分析麵包圖片，辨識出它的種類

## Installation
你需要先安裝以下套件
* keras
* sklearn
* opencv
* matplotlib

## Usage
1. 先在`breads/training`放入要訓練的麵包種類，以及在`breads/validation`放入訓練集10%資料量的測試集
2. 運行`main.py`，會訓練出一個`vgg16_1.h5`的麵包模型
3. 運行`evaluation.py`，開始分析麵包的圖片，列出前三種符合機率最高的麵包種類
> 如果要換分析的麵包圖的話，修改evaluation.py內的img_path變數