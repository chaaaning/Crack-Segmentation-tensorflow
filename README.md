- [1. 개요](#1-개요)
- [2. Training Environment](#2-training-environment)
- [3. Model](#3-model)
  - [3.1 Structure of code and dataset](#31-structure-of-code-and-dataset)
    - [3.1.1 Code](#311-code)
    - [3.1.2 Dataset](#312-dataset)
    - [3.1.3 Model Weights 파일](#313-model-weights-파일)
  - [3.2 Parameter](#32-parameter)
  - [3.3 모델의 입출력](#33-모델의-입출력)
- [4. How To Use](#4-how-to-use)
  - [4.1 환경 세팅](#41-환경-세팅)
  - [4.2 Train](#42-train)
  - [4.3 Test](#43-test)
  - [4.4 Inference](#44-inference)
    - [4.4.1 Image \& Video File Inference](#441-image--video-file-inference)
    - [4.4.2 Use Dashboard](#442-use-dashboard)
  - [4.5 Quantized tensorflow lite mod](#45-quantized-tensorflow-lite-mod)
- [5. 시연 영상](#5-시연-영상)
  - [5.1 Inference in Terminal](#51-inference-in-terminal)
  - [5.2 Mobile Inference](#52-mobile-inference)
  - [5.3 Streamlit Dashboard](#53-streamlit-dashboard)

## 1. 개요
- AI hub에 구축되어 있는 [도로장애물/표면 인지 영상(수도권)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=179) 데이터 셋을 이용하여 자동으로 도로의 크랙을 탐지할 수 있는 모델을 생성하고, 이를 모바일 환경에서 작동할 수 있도록 경량화한 모델을 개발함
- `MobilnetV2` 기반 `UNet` 모델을 `QAT(Quantization Aware Training)`방식으로 학습을 수행하고, `Dynamic Quantization`을 사용하여 경량화된 모델을 개발함
  
## 2. Training Environment
|Name|Version|
|:-------------------:|:---------------------------|
|GPU                  |Nvidia RTX 3060 D6 12GB     |
|Nvidia Driver Version|511.23                      | 
|CUDA                 |11.1                        |
|Python               |3.9.13                      |
|Framework            |Tensorflow-gpu(v2.7.0)      |

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
│   └── inference_data
│       └──images 
├── requirements.txt
├── environment.yml
├── result
│   ├── pred
│   ├── true
│   └── image<or video>_inferences  //inference.py 실행 시 생성 
└── 매뉴얼.ipynb
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
|streamlit_dashboard.py|streamlit을 이용하여 데이터를 웹에서 시각화해줌|


#### 3.1.2 Dataset
- `.txt`로 생성된 dummy file은 삭제 후 각 경로에 알맞는 데이터를 삽입하여 사용

|Name|Description                                  |
|:--------:|:--------------------------------------|
|train     |학습 이미지 데이터셋                    | 
|train_anno|학습 이미지 데이터의 annotation 정보    |
|test      |테스트 이미지 데이터셋                  |
|test_anno |테스트 이미지 데이터셋의 annotation 정보|
|inference_data|추론하고자 하는 데이터(images 경로)와 json 파일을 input하는 경로|

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
- 설치하고자 하는 환경에 anaconda가 깔려있음을 전제로 함
```bash
# 1. 기존의 가상환경 사용
conda activate <가상환경 이름>

# 2. 새로운 가상환경 생성
# 2.1 가상환경 생성
conda create -n <원하는 가상환경 이름> python=3.9

# 두 가지 방법으로 필요한 패키지를 설치할 수 있음
# 2.1.1 requirements.txt 파일을 이용하여 필요한 패키지 설치
pip install -r requirements.txt

# 2.1.2 yaml파일을 이용하여 패키지 설치
#   - environment.yml 파일에서 name(맨위)과 prefix(맨아래)를 본인의 가상환경 이름에 맞게 변경해주세요(어떠한 형식으로 변경하는지는 해당 파일을 참조해주세요)

#    가상환경이 activation 되어 있을 때
conda env update --file environment.yml
#    가상환경이 activation 되어 있지 않을 때
conda env update --name <가상환경이름> --file environment.yml

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
    --is_DFsave : 추론 결과를 pd.DataFrame 형태로 저장 여부를 지정 (json 파일이 탐색되면 True로 자동 설정)
    --is_quantize : tf.lite를 활용한 모델을 사용할 경우 True로 지정
    --file_type : image와 video 중 하나를 지정해야 하므로 image 지정
    --frame : video inference 시 frame을 지정
'''

python inference.py --model UNet --base_model MobileNetV2 --crop_height 288 --crop_width 384 --num_classes 2 --weights "./weights/UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5" --file_type image
```
- 학습된 모델을 이용하여 대전시 도로 영상 객체 인식 데이터셋에 대한 inference 함
- 추론하고자 하는 데이터는 `root/data/inference_data/images` 경로에 이미지 파일을 input 함
- `root/data/inference_data/images`의 하위 디렉토리는 몇 단계가 이어지더라도 상관없으나, 이미지 파일만 있어야 함 (`jpg`, `jpeg`, `png` 등)
- 추론 결과는 `root/result/<file_type 입력값>_inferences`에 저장함 (`file_type`이 image이면 `root/result/image_inferences`에 저장, 해당 경로가 없다면 자동 생성됨)
- `json`파일과 연계하는 추론을 위해서 `root/data/inferece_data` 하위에 `.json` 형태로 저장하면 알아서 탐색하여 실행됨
- `is_DFsave` 옵션은 추론 결과에 대한 내용을 `pd.DataFrame`형태로 저장하지만, `json` 파일이 탐색되면 자동으로 `True`로 설정됨
- `json`파일이 없더라도, `is_DFsave`옵션을 통해 masking 비율 정보를 저장할 수 있음 (`is_DFsave`를 지정하지 않으면 추론 이미지만을 저장)
- 다른 이미지, 비디오 데이터에 대해 추론 가능하나, 대전시 도로 영상 객체 인식 데이터 셋에 대해 추론 결과를 이미지로 저장하고 Making Ratio를 계산하여 이미지에 추론 결과를 `.csv`파일로 저장함
- 이 때, 마스킹 정보는 `mask_indptr`과 `mask_indices`에 따로 저장하여 `csr_matrix`형식으로 불러 올 수 있음 (load 시 `scipy.sparse.csr_matrix` 활용 권장)
- 추론 시, 중간에 멈추고 싶다면 `Keyboard Interrupt`를 발생시켜, 발생 직전까지 결과를 저장함
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


### 4.5 Quantized tensorflow lite mod
- 학습된 tensorflow model을 모바일에 넣기 위해서 Quantied Tensorflow lite 모델로 변환함
- 변환된 tflite 모델에 모델의 metadata를 기록해주어야 함(class 정보, normalization 정보 등)
- 아래의 코드는 `quantization.ipynb` 파일에 정리되어 있으니 자세한 내용은 해당 파일 참조

## 5. 시연 영상
### 5.1 Inference in Terminal
![terminal inference](./assets/inference_demo.gif)
### 5.2 Mobile Inference
![mobile inference](./assets/road_crack_detection.gif)
### 5.3 Streamlit Dashboard
![streamlit dashboard](./assets/streamlit_demo.gif)

