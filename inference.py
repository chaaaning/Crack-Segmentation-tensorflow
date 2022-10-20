"""
The file defines the predict process of a single RGB image.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.helpers import check_related_path
from utils.utils import load_image, decode_one_hot
from utils.helpers import get_dataset_info
from keras_applications import imagenet_utils
from builders import builder
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import argparse
import sys
import cv2
import os

print("progress to start ...")

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, required=True)
parser.add_argument('--base_model', help='Choose the backbone model.', type=str, default=None)
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, required=True)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)
parser.add_argument('--image_path', help='The path of to enable image.', type=str, required=True)
parser.add_argument('--save_path', help='The path of predicted image.', type=str, default=os.path.join(os.getcwd(), 'predictions'))
parser.add_argument('--json_path', help='The path of to load json.', type=str, default=None)
parser.add_argument('--is_save', help='save options.', type=bool, default=False)

args = parser.parse_args()

### 필요한 함수 정의 ###
# 1. image와 prediction 합치기
def merge_img(frame, pred, img_size):
    re_frm = cv2.resize(frame, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    add_mask = re_frm.copy()
    add_mask[pred[:,:]!=0]=[0,255,0]
    return add_mask

# 2. 추가 데이터 수집 로직을 위한 json Load
def json_to_df(json_path, save=False):
    with open(json_path, "r") as f:
        json_data = json.load(f)
        
    # 컬럼명을 정리하고 싶다면,
    key_list = ['id', 'path', 'size', 'name', 'format', 'timestamp', 'record_time',
            'latitude', 'longitude', 'countRegions', 'parentRealName', 'assetTags',  
            'predicted_by_api']
    
    # 유동적으로 하고 싶다면,
    # check_key_set = set()
    # for i in range(len(json_data["assets"])):
    #     check_key_set = check_key_set.union(set(json_data["assets"][i]["image"].keys()))
    # key_list = list(check_key_set)
    to_make_pd_dict = dict()

    for i in range(len(json_data["assets"])):
        if i==0:
            for key in key_list:
                try:
                    to_make_pd_dict[key] = [json_data["assets"][i]["image"][key]]
                except:
                    to_make_pd_dict[key] = [np.nan]
        else:
            for key in key_list:
                try:
                    to_make_pd_dict[key].append(json_data["assets"][i]["image"][key])
                except:
                    to_make_pd_dict[key].append(np.nan)
                    
    json_df = pd.DataFrame(to_make_pd_dict)
    
    if save:
        json_df.to_csv(os.path.basename(json_path).rstrip(".json")+".csv", index=False)
        
    return json_df     
#######################

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

# load weights
print('Loading the weights...')
if args.weights is not None:
    print(os.getcwd())
    net.load_weights(filepath=args.weights)
else:
    if not os.path.exists(args.weights):
        raise ValueError('The weights file does not exist in \'{path}\''.format(path=args.weights))
    net.load_weights(args.weights)

# begin testing
print("\n***** Begin testing *****")
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", args.num_classes)

print("")

# load_images & json
df_json = json_to_df(args.json_path)
image_names = df_json["name"]+"."+df_json["format"]
image_names = image_names.apply(lambda x: os.path.basename(x))

# make ETL pipeline initialization -> output json
# 들어 가야할 내용 : 'path', 'name', 'format', 'latitude', 'longitude'
# key_list = ['path', 'name', 'format', 'latitude', 'longitude']
# data_extract = dict()
# for key in key_list:    # dictionary 초기화 (json을 어떻게 넘길까 고민,,)
#     data_extract[key] = []

# predict & save
for i, img_name in enumerate(image_names):
    sys.stdout.write('\rRunning test image %d / %d'%(i+1, len(image_names)))
    sys.stdout.flush()

    # loading image & convert dim
    img_pname = args.image_path+"\\"+img_name.split(".")[0]+"\\"+img_name
    init_image = cv2.resize(load_image(img_pname), dsize=(args.crop_width, args.crop_height))
    image = imagenet_utils.preprocess_input(init_image.astype('float32'), data_format='channels_last', mode='torch')

    if np.ndim(image) == 3:
        image = np.expand_dims(image, axis=0)
    assert np.ndim(image) == 4

    preds = net.predict(image)
    preds = tf.nn.softmax(preds)
    
    if np.ndim(preds) == 4:
        prediction = np.squeeze(preds, axis=0)
    
    np_prediction = decode_one_hot(prediction)
    prediction = Image.fromarray(np.uint8(prediction))
    
    # save the prediction
    if args.is_save:
        if i==0:
            _, file_name = os.path.split(img_name)
            if os.path.exists(args.save_path):
                save_path = args.save_path
            else:
                save_path = os.mkdir(os.path.join(os.getcwd(), 'predictions'))
                print("make prediction save path !")
        ## Save(1) - PIL save
        # prediction.save(os.path.join(save_path, img_name))
        
        ## Save(2) - pyplot save
        # plt.imshow(prediction)
        # plt.savefig(os.path.join(save_path, img_name))
        
        ## Save(3) - Image merge save
        img_mask_merge = merge_img(init_image, np_prediction, (args.crop_width, args.crop_height))
        cv2.imwrite(os.path.join(save_path, img_name), img_mask_merge)
        