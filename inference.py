from utils.utils import load_image, decode_one_hot
from keras_applications import imagenet_utils
from builders import builder
from scipy.sparse import csr_matrix
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import argparse
import sys
import cv2
import os
from tqdm import tqdm

print("progress to start ...")

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, required=True)
parser.add_argument('--base_model', help='Choose the backbone model.', type=str, default=None)
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, required=True)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)
parser.add_argument('--input_path', help='The path of to enable image.', type=str, required=True)
parser.add_argument('--img_save_path', help='The path of predicted image.', type=str, default=os.path.join(os.getcwd(), 'image_predictions'))
parser.add_argument('--vdo_save_path', help='The path of predicted video.', type=str, default=os.path.join(os.getcwd(), 'video_predictions'))
parser.add_argument('--json_path', help='The path of to load json.', type=str, default=None)
parser.add_argument('--is_save', help='save options.', type=bool, default=False)
parser.add_argument('--is_quantize', help='Input quantize T or F.', type=bool, default=True)
parser.add_argument('--file_type', help='choose image or video', type=str, choices=["image", "video"], required=True)
parser.add_argument('--frame', help='If you input video file, need frame', type=int, default=30)

args = parser.parse_args()

### 필요한 함수 정의 ###
# -- 1. image와 prediction 합치기
def merge_img(frame, pred, img_size, is_BGR=True):
    re_frm = cv2.resize(frame, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    if is_BGR:
        add_mask = cv2.cvtColor(re_frm, cv2.COLOR_BGR2RGB)
    else:
        add_mask = re_frm.copy()
    add_mask[pred[:,:]!=0]=[0,255,0]
    return add_mask

## -- 2. predict frame or img
def im_pred(init_image, pred_net):
    image = imagenet_utils.preprocess_input(init_image.astype('float32'), data_format='channels_last', mode='torch')

    if np.ndim(image) == 3:
        image = np.expand_dims(image, axis=0)
    assert np.ndim(image) == 4

    preds = pred_net.predict(image)
    preds = tf.nn.softmax(preds)
    
    if np.ndim(preds) == 4:
        prediction = np.squeeze(preds, axis=0)
    
    np_prediction = decode_one_hot(prediction)
    
    return np_prediction
    
# -- 3. 추가 데이터 수집 로직을 위한 json Load
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

# -- 4. masking rate calculate    
def calc_masking_rate(arr, width, height):
    masking_val = arr.sum()
    return (masking_val*100)/(width*height)

#######################

# quatization
quantize_model = tfmot.quantization.keras.quantize_model

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

if args.is_quantize:
    net = quantize_model(net)

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

#### inference file이 image 일 때
if args.file_type=="image":
    ## Keyboard Interrupt가 발생하면 저장할 수 있도록 코드 라인 (try-except) 형성
    try:
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

        mask_indptr, mask_indices, masking_rates = [], [], []
        # predict & save
        for i, img_name in enumerate(image_names):

            # loading image & convert dim
            img_pname = args.input_path+"\\"+img_name.split(".")[0]+"\\"+img_name
            init_image = cv2.resize(load_image(img_pname), dsize=(args.crop_width, args.crop_height))
            prediction = im_pred(init_image, net)
            
            csr_mat = csr_matrix(prediction)
            mask_indptr.append(csr_mat.indptr.tolist())
            mask_indices.append(csr_mat.indices.tolist())
            masking_rates.append(calc_masking_rate(prediction, args.crop_width, args.crop_height))
            
            # save the prediction
            if args.is_save:
                if i==0:
                    # _, file_name = os.path.split(img_name)
                    if os.path.exists(args.img_save_path):
                        save_path = args.img_save_path
                    else:
                        save_path = os.path.join(os.getcwd(), 'image_predictions')
                        os.mkdir(save_path)
                        print("make image prediction save path !")

                img_mask_merge = merge_img(init_image, prediction, (args.crop_width, args.crop_height))
                cv2.imwrite(os.path.join(save_path, img_name), img_mask_merge) 
            
            sys.stdout.write('\rRunning test image %d / %d'%(i+1, len(image_names)))
            sys.stdout.flush()
            
        save_df = df_json.iloc[:,:].copy()
        save_df["mask_indptr"] = mask_indptr
        save_df["mask_indices"] = mask_indices
        save_df["masking_rate"] = masking_rates
        save_df.to_csv(os.path.join(save_path, "masking_result.csv"), index=False)
        
            
    except:
        img_mask_merge = merge_img(init_image, prediction, (args.crop_width, args.crop_height))
        cv2.imwrite(os.path.join(save_path, img_name), img_mask_merge)
        
        print("")
        print(f"Interrupt !! save result data ...")
        save_df = df_json.iloc[:i,:].copy()
        save_df["mask_indptr"] = mask_indptr
        save_df["mask_indices"] = mask_indices
        save_df["masking_rate"] = masking_rates
        save_df.to_csv(os.path.join(save_path, "masking_result.csv"), index=False)
        print(f"Saved Result CSV !!!")

#### inference file이 video 일 때        
else:
    try:
        print("Inference Video ...")
        for video_name in tqdm((os.listdir(args.input_path))):
            capture = cv2.VideoCapture(os.path.join(args.input_path, video_name))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.crop_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.crop_height)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            if args.is_save:
                if os.path.exists(args.vdo_save_path):
                    save_path = args.vdo_save_path
                else:
                    save_path = os.path.join(os.getcwd(), 'video_predictions')
                    os.mkdir(save_path)
                    print("make video save path !")
                out = cv2.VideoWriter(os.path.join(save_path, video_name), fourcc, args.frame, (args.crop_width, args.crop_height))
            
            
            while cv2.waitKey(33) < 0:
                full_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                ret, frame = capture.read()
                init_frame = cv2.resize(frame, dsize=(args.crop_width, args.crop_height))
                if not ret:
                    break

                pred_img = im_pred(init_frame, net)
                convert_img = merge_img(init_frame, pred_img, (args.crop_width, args.crop_height), False)
                
                cv2.imshow("Test Vidoe Crack Detect", convert_img)
                if args.is_save:
                    out.write(convert_img)

                capture.release()
                if args.is_save:
                    out.release()
                    print("Inference SAVE !!!")
                cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
            if args.is_save:
                print("Keyboard Interrupt !")
                out.release()
                print("Saved Current Video!")
            cv2.destroyAllWindows()