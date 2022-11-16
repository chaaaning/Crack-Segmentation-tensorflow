## 1. 개요
- AI hub에 구축되어 있는 [도로장애물/표면 인지 영상(수도권)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=179) 데이터 셋을 이용하여 자동으로 도로의 크랙을 탐지할 수 있는 모델을 생성하고, 이를 모바일 환경에서 작동할 수 있도록 경량화한 모델을 개발함
- `MobilnetV2` 기반 `UNet` 모델을 `QAT(Quantization Aware Training)`방식으로 학습을 수행하고, `Dynamic Quantization`을 사용하여 경량화된 모델을 개발함
  
## 2. Training Environment
|||
|:-------:|:---------------------------|
|GPU      |Nvidia RTX 3060 D6 12GB     | 
|Python   |3.9.13                      |
|Framework|Tensorflow-gpu(v2.7.0)      |
|CUDA     |11.1                        |
|||

## 3. Model
### 3.1 Structure of code and dataset
- 학습의 관한 코드 및 데이터는 다음과 같이 이루어져 있음
```bash
├── code
│   ├── base_models
│   ├── builders
│   ├── quantization.ipynb
│   ├── data_list
│   ├── eval.py
│   ├── inference.py
│   ├── label.txt
│   ├── models
│   ├── predictions
│   ├── test.py
│   ├── tflite_models
│   ├── train.py
│   ├── utils
│   └── weights
├── data
│   ├── test
│   ├── test_anno
│   ├── train
│   └── train_anno
├── requirements.txt
└── result
    ├── pred
    └── true
```
#### 3.1.1 Code
- [Amazing-Semantic-Segmentation](https://github.com/luyanger1799/Amazing-Semantic-Segmentation#readme) 코드를 참조하여 학습을 진행함
  
|Name|Description                                             |
|:------------:|:---------------------------------------------|
|builders      |모델의 구조를 load                            | 
|data_list     |학습 이미지 데이터 리스트 파일                |
|utils|학습에 필요한 각종 요소들이 담겨 있음(loss, data generator 등)|
|weights|학습된 가중치파일이 담겨 있음|
|base_models   |Backbon, MobileNetV2 사용                     |
|models        |Models, UNet 사용                             |
|eval.py|주어진 dataset을 평가|
|train.py|train dataset를 이용하여 모델 학습|
|test.py|학습된 모델로 test dataset 평가|
|result|test dataset을 이용하여 test 시 사용되는 원본이미지와 예측이미지가 저장돼있음|
|label.txt     |segmentation class information(background,crack)|
|quantization.ipynb|학습된 모델의 가중치를 quantization을 적용한 tflite model로 변환하는 코드|
|tflist_models|tensorflow lite모델이 저장되어 있음|
|inference.py  |학습된 모델을 이용하여 image와 video에 대한 inference service를 제공<br>      |
|predictions||
|||


#### 3.1.2 Dataset
|Name|Description                                  |
|:--------:|:--------------------------------------|
|train     |학습 이미지 데이터셋                    | 
|train_anno|학습 이미지 데이터의 annotation 정보    |
|test      |테스트 이미지 데이터셋                  |
|test_anno |테스트 이미지 데이터셋의 annotation 정보|

### 3.2 Parameter
- 다음과 같은 조건으로 parameter를 지정하여 학습을 진행함
  
|Name|Value                                  |
|:--------:|:--------------------------------------|
|Loss     |Categorical Cross-Entropy               | 
|Batch Size|16    |
|Learning Rate|1e-3|
|Learning Rate Decay|Cosine decay(per batch)|
|Optimizer|Adam Optimizer|
|Image size|384x288                  |
|Number of class |2|
|Validation Ratio|20%|

## 4. How To Use
### 4.1 환경 세팅
- 현재 서버에 `tensorflow` 라는 가상환경을 생성한 상태이며 이것을 이용하여 진행해도 무방함.
```bash
# 1. 기존의 가상환경 사용
conda activate tensorflow

# 2. 새로운 가상환경 생성
#  2-1. 가상환경 생성
conda create -n <원하는 가상환경 이름> python=3.9

#  2-2 필요한 패키지 설치
pip install -r requirements.txt
```

### 4.2 Train
```bash
'''
:param 
    --model : Unet 모델을 사용
    --base_model : backbone model로 MobileNetV2 사용
    --loss : Categorical Cross-Entropy loss 사용
    --batch_size : train dataset의 batch size 16으로 지정
    --valid_batch_size : validation dataset의 batch size 16으로 지정
    --crop_height : 이미지의 세로사이즈를 288로 지정
    --crop_width : 이미지의 가로사이즈를 384로 지정
    --num_epochs : 학습 Epoch을 20으로 지정
    --dataset : train dataset의 경로를 지정
    --random_seed : train dataset을 한번 섞어주는데, 이때 항상 동일한 순서로 섞이도록 seed를 80으로 지정
    --num_classes : segmentation의 분류 클래스 갯수를 2로 지정
    --learning_rate : 초기학습률을 1e-3로 지정
'''
python train.py --model UNet --base_model MobileNetV2 --loss ce --batch_size 16 --valid_batch_size 16 --crop_height 288 --crop_width 384 --num_epochs 20 --dataset ../data/train --random_seed 80 --num_classes 2 --learning_rate 1e-3
```
- 매 epoch마다 `weight`폴더에 학습된 모델의 가중치가 저장됨
- 1epoch당 2번의 validation을 진행(wandb를 이용하여 기록을 진행했으나 제출코드에서는 제거함)
  
### 4.3 Test
```bash
'''
:param 
    --model : Unet 모델을 사용
    --base_model : backbone model로 MobileNetV2 사용
    --batch_size : test dataset의 batch size 16으로 지정
    --crop_height : 이미지의 세로사이즈를 288로 지정
    --crop_width : 이미지의 가로사이즈를 384로 지정
    --dataset : test dataset의 경로를 지정
    --num_classes : segmentation의 분류 클래스 갯수를 2로 지정
    --weights : load할 가중치파일의 경로를 지정
    --isQAT : QAT(Quantization Aware Training)을 적용한 모델이므로 yes를 지정
'''

python test.py --model UNet --base_model MobileNetV2 --batch_size 16 --crop_height 288 --crop_width 384 --weight ./weights/UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5 --dataset ../data/test --num_classes 2 --isQAT yes
```
- 학습된 모델을 이용하여 test dataset을 test함
- 테스트 결과로 모든 배치의 평균적인 dice score를 보여주고, 원본 이미지와 예측이미지를 `result` 폴더에 저장함

### 4.4 Inference
- 내용 작성 부탁드립니다

### 4.5 Quantized tensorflow lite model in Mobile
- 학습된 tensorflow model을 모바일에 넣기 위해서 Quantied Tensorflow lite 모델로 변환함
- 변환된 tflite 모델에 모델의 metadata를 기록해주어야 함(class 정보, normalization 정보 등)
- 아래의 코드는 `quantization.ipynb` 파일에 정리되어 있으니 자세한 내용은 해당 파일 참조 부탁드립니다.



