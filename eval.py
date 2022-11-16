import tensorflow as tf
from tqdm import tqdm
# from keras_metrics import f1_score
from utils.metrics import F1Score
from operator import truediv
from keras import backend as K
import numpy as np
from utils.losses import dice_coef
from PIL import Image

def evaluate(net, dataloader,isTrain=True):
    num_val_batches = len(dataloader)
    n_classes = 2

    score = 0
    batches = 0

    for images, masks in tqdm(dataloader,total=num_val_batches,desc='Validation round',unit='batch',leave=False):
        preds = tf.argmax(tf.nn.softmax(net(images),axis=-1),axis=-1).numpy().astype(np.float32)
        masks = masks[...,1].astype(np.float32)

        # test일 경우 실제, 예측 이미지 저장
        if isTrain == False:
            # imagenet
            mean = [0.485,0.456,0.406]
            std=[0.229,0.224,0.225]

            denormalized_true_image = ((images[0]*std)+mean)*255.0
            Image.fromarray(denormalized_true_image.astype(np.uint8)).save(f'../result/true/true_{batches}.jpg')
            Image.fromarray((preds[0]*255).astype(np.uint8)).save(f'../result/pred/pred_{batches}.jpg')
        
        score += dice_coef(masks,preds)

        batches += 1
        if batches >= num_val_batches:
            break

    print(f"\nDice-Score : {score/num_val_batches}\n")
    # validation set에 대한 평균(배치별) dice score return
    return score / num_val_batches

