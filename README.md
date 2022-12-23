- [1. 개요](#1-개요)
- [2. Training Environment](#2-training-environment)
- [3. Model](#3-model)
    - [3.1. Structure of code and dataset](#31-structure-of-code-and-dataset)
        - [3.1.1. code ](#311-code)
        - [3.1.2. dataset](#312-dataset)
        - [3.1.3. model weights file](#313-model-weights-파일)
    - [3.2. Parameter](#32-parameter)
    - [3.3. Model Input/Outplut](#33-모델의-입출력)
- [4. How to use](#4-how-to-use)
    - [4.1. Setting Envirionment](#41-환경-세팅)
    - [4.2. Train](#42-train)
    - [4.3. Test](#43-test)
    - [4.4. Inference](#44-inference)
        - [4.4.1. Image/Video File Inference](#441-image--video-file-inference)
        - [4.4.2. Use Dashboard](#442-use-dashboard)
    - [4.5. Quantized tensorflow lite model in Mobile](#45-quantized-tensorflow-lite-model-in-mobile)

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

## 3. Model
### 3.1 Structure of code and dataset
- 학습의 관한 코드 및 데이터는 다음과 같이 이루어져 있음
```bash
├── README.md
├── code
│   ├── README.md
│   ├── base_models
│   ├── builders
│   ├── checkpoints
│   ├── data_list
│   ├── eval.py
│   ├── image_predictions
│   ├── inference.py
│   ├── label.txt
│   ├── models
│   ├── predictions
│   ├── quantization.ipynb
│   ├── streamlit_dashboard.py
│   ├── test.py
│   ├── tflite_models
│   ├── train.py
│   ├── utils
│   └── weights
├── data
│   ├── test
│   ├── test_anno
│   ├── train
│   ├── train_anno
│   └── 대전시_도로_영상_객체_인식_데이터셋_2020_위치정보
├── requirements.txt
├── result
│   ├── pred
│   └── true
└── 매뉴얼.txt
```

#### 3.1.1 Code
- [Amazing-Semantic-Segmentation](https://github.com/luyanger1799/Amazing-Semantic-Segmentation#readme) 코드를 참조하여 학습을 진행함
  
|Name|Description                                                                           |
|:------------:|:---------------------------------------------------------------------------|
|builders      |모델의 구조를 load                                                           |  
|data_list     |학습 이미지 데이터 리스트 파일                                               |
|utils         |학습에 필요한 각종 요소들이 담겨 있음(loss, data generator 등)               |
|weights       |학습된 가중치파일이 담겨 있음                                                |
|base_models   |Backbon, MobileNetV2 사용                                                    |
|models        |Models, UNet 사용                                                            |
|eval.py       |주어진 dataset을 평가                                                        |
|train.py|train dataset를 이용하여 모델 학습                                                 |
|test.py|학습된 모델로 test dataset 평가                                                     |
|result|test dataset을 이용하여 test 시 사용되는 원본이미지와 예측이미지가 저장돼있음         |
|label.txt     |segmentation class information(background,crack)                            |
|quantization.ipynb|학습된 모델의 가중치를 quantization을 적용한 tflite model로 변환하는 코드|
|tflist_models|tensorflow lite모델이 저장되어 있음                                           |
|inference.py  |학습된 모델을 이용하여 image와 video에 대한 inference service를 제공         |
|image_predictions|inference.py 실행 결과가 저장됨|
|streamlit_dashboard.py|streamlit을 이용하여 데이터를 웹에서 시각화해줌|


#### 3.1.2 Dataset
|Name|Description                                  |
|:--------:|:--------------------------------------|
|train     |학습 이미지 데이터셋                    | 
|train_anno|학습 이미지 데이터의 annotation 정보    |
|test      |테스트 이미지 데이터셋                  |
|test_anno |테스트 이미지 데이터셋의 annotation 정보|
|대전시_도로_영상_객체_인식_데이터셋_2020_위치정보|대전시 도로 이미지 및 관련 json 파일|

#### 3.1.3 Model Weights 파일
```bash
├── DeepLabV3Plus_based_on_MobileNetV2_29.h5
├── PAN_based_on_MobileNetV2_29.h5
├── UNet_based_on_MobileNetV2_CE.h5
├── UNet_based_on_MobileNetV2_CE_QAT_288384.h5
├── UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5
├── UNet_based_on_MobileNetV2_CE_QAT_288384_aug.h5
├── UNet_based_on_MobileNetV2_QAT.h5
├── UNet_based_on_MobileNetV2_gamma_2.h5
├── UNet_based_on_MobileNetV2_gamma_5.h5
└── UNet_based_on_VGG16_29.h5
```
- `UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5` 파일이 최종 생성 모델 파일임

### 3.2 Parameter
- 다음과 같은 조건으로 parameter를 지정하여 학습을 진행함
  
|Name|Value                                        |
|:--------:|:--------------------------------------|
|Loss     |Categorical Cross-Entropy               | 
|Batch Size|16                                     |
|Learning Rate|1e-3                                |
|Learning Rate Decay|Cosine decay(per batch)       |
|Optimizer|Adam Optimizer                          |
|Image size|384x288                                |
|Number of class |2                                |
|Validation Ratio|10%                              |

### 3.3 모델의 입출력
|Type      |Description                            |Shape            |Data Type|
|:--------:|:--------------------------------------|:---------------:|:-------:|
|Input      |MinMax Scaling을 취한 후, ImageNet dataset 기준 mean, std를 이용하여 normalization을 진행한 Tensor  |(batch size,288,384,3)|float32|
|Output    |Tensor(logits 형태)                    | (batch size,288,384,2)|float32|

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
# 모델별 실험
# UNet
python test.py --model UNet --base_model MobileNetV2 --batch_size 32 --weight ./weights/UNet_based_on_MobileNetV2_CE.h5 --dataset ../data/test --num_classes 2

# PAN
python test.py --model PAN --base_model MobileNetV2 --batch_size 32 --weight ./weights/PAN_based_on_MobileNetV2_29.h5 --dataset ../data/test --num_classes 2

# DeepLabV3Plus
python test.py --model DeepLabV3Plus --base_model MobileNetV2 --batch_size 32 --weight ./weights/DeepLabV3Plus_based_on_MobileNetV2_29.h5 --dataset ../data/test --num_classes 2

# Base 모델별 실험
# MobileNetV2
python test.py --model UNet --base_model MobileNetV2 --batch_size 32 --weight ./weights/UNet_based_on_MobileNetV2_CE.h5 --dataset ../data/test --num_classes 2

# VGG16
python test.py --model UNet --base_model VGG16 --batch_size 16 --weight ./weights/UNet_based_on_VGG16_29.h5 --dataset ../data/test --num_classes 2

# 최종 모델 실험
python test.py --model UNet --base_model MobileNetV2 --batch_size 16 --crop_height 288 --crop_width 384 --weight ./weights/UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5 --dataset ../data/test --num_classes 2 --isQAT yes
```
- 학습된 모델을 이용하여 test dataset을 test함
- 테스트 결과로 모든 배치의 평균적인 dice score를 보여주고, 원본 이미지와 예측이미지를 `result` 폴더에 저장함

### 4.4 Inference
#### 4.4.1 Image & Video File Inference
```python
'''
:param 
    --model : Unet 모델을 사용
    --base_model : backbone model로 MobileNetV2 사용
    --batch_size : test dataset의 batch size 16으로 지정
    --crop_height : 이미지의 세로사이즈를 288로 지정
    --crop_width : 이미지의 가로사이즈를 384로 지정
    --dataset : test dataset의 경로를 지정
    --num_classes : segmentation의 분류 클래스 갯수를 2로 지정
    --weights : 학습 모델의 성능이 가장 좋은 모델 weight를 load
    --input_path : 추론에 활용될 data set (대전시 도로 영상 객체 인식) 경로 지정
    --img_save_path : image inference save_path, default 시 현재 경로 생성
    --vdo_save_path : video inference save_path, default 시 현재 경로 생성
    --json_path : 추론에 활용되는 data set (대전시 도로 영상 객체 인식)의 정보를 담는 json 파일의 경로 지정
    --is_DFsave : 추론 결과를 pd.DataFrame 형태로 저장 여부를 지정 (json_path 인자 입력 시 True로 자동 설정)
    --is_quantize : tf.lite를 활용한 모델을 사용할 경우 True로 지정
    --file_type : image와 video 중 하나를 지정해야 하므로 image 지정
    --frame : video inference 시 frame을 지정
'''

python inference.py --model UNet --base_model MobileNetV2 --crop_height 288 --crop_width 384 --num_classes 2 --weights "./weights/UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5" --input_path "../data/대전시_도로_영상_객체_인식_데이터셋_2020_위치정보/images" --json_path "../data/대전시_도로_영상_객체_인식_데이터셋_2020_위치정보/DataSetIII.json" --file_type image
```
- 학습된 모델을 이용하여 대전시 도로 영상 객체 인식 데이터셋에 대한 inference 함
- inference에 활용할 데이터는 `root/data` 디렉토리에 미리 저장하고, 그에 맞게끔 `input_path`를 지정
- `input_path`의 하위 디렉토리는 몇 단계가 이어지더라도 상관없으나, 이미지 파일만 있어야 함 (`jpg`, `jpeg`, `png` 등)
- 추론 결과는 `./<file_type 입력값>_predictions`에 저장함 (`file_type`이 image이면 `image_predictions`에 저장)
- `is_DFsave` 옵션은 추론 결과에 대한 내용을 `pd.DataFrame`형태로 저장하지만, `json_path`가 입력되면 자동으로 `True`로 설정됨
- `json_path`는 `root/data`의 하위 디렉토리에 `json`파일을 저장하고, 절대 경로의 형태로 입력
- `json_path`가 입력되지 않더라도, `is_DFsave`옵션을 통해 masking 비율 정보를 저장할 수 있음 (`is_DFsave`를 지정하지 않으면 추론 이미지만을 저장)
- 다른 이미지, 비디오 데이터에 대해 추론 가능하나, 대전시 도로 영상 객체 인식 데이터 셋에 대해 추론 결과를 이미지로 저장하고 Making Ratio를 계산하여 이미지에 추론 결과를 `.csv`파일로 저장함
- 이 때, 마스킹 정보는 `mask_indptr`과 `mask_indices`에 따로 저장하여 `csr_matrix`형식으로 불러 올 수 있음 (load 시 `scipy.sparse.csr_matrix` 활용 권장)
​
#### 4.4.2 Use Dashboard
```python
'''
# 권장 설치 라이브러리
    pip install streamlit
    pip install pydeck
    pip install seaborn
'''
streamlit run streamlit_dashboard.py --browser.serverAddress localhost
```
- `streamlit_dashboard`에서 제공하는 정보는 운용 환경, requirements, 추론 결과 DataFrame, histogram, boxplot, heatmap 이 있음
- 현재 버전은 [kisti의 도로 영상 객체 데이터셋](https://aida.kisti.re.kr/data/4fa7657c-5d21-4a20-bca9-ca53ac4c85e4)을 기준으로 만들어 졌기 때문에 해당 데이터 셋의 결과에 맞게 끔 최적화 되어 있음
- [streamlit 공식문서](https://docs.streamlit.io/)를 참조하여 추가적인 기능을 생성할 수 있음


### 4.5 Quantized tensorflow lite model in Mobile
- 학습된 tensorflow model을 모바일에 넣기 위해서 Quantied Tensorflow lite 모델로 변환함
- 변환된 tflite 모델에 모델의 metadata를 기록해주어야 함(class 정보, normalization 정보 등)
- 아래의 코드는 `quantization.ipynb` 파일에 정리되어 있으니 자세한 내용은 해당 파일 참조 부탁드립니다.



