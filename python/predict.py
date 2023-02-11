# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : predict.py

import  os
import tensorflow as tf
from utils.conf_utils import get_predict_conf
from utils.data_utils import get_predict_data
from models.joint import DNNClassifier

def prob2rate(prob,days):
    """
    :param prob: [n_sample, 31] 前30列为每月30日内（索引0~29）行为发生的概率分布，第31列为行为不发生的概率
    :param days: [n_sample, 1] 样本当前已经历的天（-1~29），-1表示当前期次还未到，0表示处在第1天
    :return:[n_sample,1]
    根据行为发生的概率分布prob与已经历的天days，计算样本已过n日行为仍未发生，但在未来仍然可能发生的概率
    """
    if prob.shape[0] != days.shape[0]:
        raise ValueError("the sample num of pred and days are not equal!")
    if prob.shape[1] != 31:
        raise ValueError("Unexpected pred shape[1] %d, expect 31" % (prob.shape[1]))

    days_mask = []
    for i in range(31):
        days_mask.append(i > days)
    days_mask = tf.cast(tf.concat(days_mask, axis=1), tf.int32) # [n_sample, 31]

    numerator = tf.reduce_sum(prob[:,:-1]*days_mask[:,:-1],axis=1,keepdims=True) #[n_sample, 30]*[n_sample, 30]->[n_sample, 1]
    denominator = tf.reduce_sum(prob*days_mask,axis=1,keepdims=True) #[n_sample, 1]
    rate = tf.divide(numerator,denominator)
    return rate


def estimate_rate(pred_data, model):
    """
    :param pred_data:
    :param model:
    :return:
    """
    X, days = pred_data
    prob = model.predict(X)
    rate = prob2rate(prob,days)
    return rate

if __name__ == "__main__":
    estimate_order, estimate_required = get_predict_conf()
    estimate_nums = len(estimate_order)
    model = DNNClassifier()
    predict_res = []
    for t, pay_info in estimate_required.items():
        for n, start_idx in pay_info.items():
            for model_name in range(start_idx,estimate_nums):
                checkpoint_path = os.path.join(os.getcwd(), 'output', 'model_files', 'checkpoint', f'{model_name}.ckpt')
                model.load_weights(checkpoint_path)
                pred_data = get_predict_data(t,n)
                predict_res.append(estimate_rate(pred_data,model))



