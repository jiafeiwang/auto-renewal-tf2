import os

from models.joint import DNNClassifier
from utils.data_utils import get_train_data
from utils.conf_utils import get_train_conf, get_feature_info

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop, Adam

if __name__ == '__main__':

    dense_feature_info, sparse_feature_info, behavior_feature_info = get_feature_info()
    train_conf = get_train_conf()

    for m, conf in train_conf.items():
        model_params = conf.get("model_param")
        hidden_units = model_params.get("hidden_units")
        ak_dim = model_params.get("ak_dim")
        dropout = model_params.get("dropout")

        for act, req in conf.get("model_required").items():
            if req == 0:
                continue
            model_name = f'{act}_{m}'
            model = DNNClassifier(dense_feature_info, sparse_feature_info, behavior_feature_info,
                                  hidden_units, ak_dim, dropout)

            optimizer = Adam(0.005)
            batch_size, epochs = 32, 30
            summary_writer_dir = os.path.join(os.getcwd(), 'output', 'tensorboard', 'callback', model_name)
            checkpoint_path = os.path.join(os.getcwd(), 'output', 'model_files', 'checkpoint', f'{model_name}.ckpt')

            X_train, y_train, X_val, y_val = get_train_data(m, act)

            # 定义checkpoint和tensorboard的回调函数
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True
                                                             )
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=summary_writer_dir)

            model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(from_logits=False),
                          metrics=[CategoricalAccuracy()])

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_val, y_val), validation_freq=1,
                      callbacks=[cp_callback, tb_callback])

            model.summary()
