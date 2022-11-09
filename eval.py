import tensorflow as tf
from tqdm import tqdm
# from keras_metrics import f1_score
from utils.metrics import F1Score
from operator import truediv
from keras import backend as K
import numpy as np
from utils.losses import dice_coef

def evaluate(net, dataloader):
    num_val_batches = len(dataloader)
    n_classes = 2

    score = 0
    batches = 0

    # 왜 자꾸 범위를 벗어나는거지....?
    # 왜 루프를 못벗어나지...?
    for images, masks in tqdm(dataloader,total=num_val_batches,desc='Validation round',unit='batch',leave=False):
        preds = tf.argmax(tf.nn.softmax(net(images),axis=-1),axis=-1).numpy().astype(np.float32)
        masks = masks[...,1].astype(np.float32)
        score += dice_coef(masks,preds)

        batches += 1
        if batches >= num_val_batches:
            break

    print(f"\nValidation Dice-Score : {score/num_val_batches}\n")
    # validation set에 대한 평균(배치별) f1 score return
    return score / num_val_batches

