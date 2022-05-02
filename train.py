from __future__ import division, print_function
import matplotlib.pyplot as plt
import glob
import matplotlib
import numpy as np
from tf_unet import unet

from utils import gen_train_batch, get_test_image


data_dir = '/home/onera'

def train():
    data_provider = gen_train_batch(data_dir)

    net = unet.Unet(channels=6, n_class=2,
            layers=3, features_root=64,
            cost_kwargs=dict(regularizer=0.001, class_weights=[1, 100]),
        )

    trainer = unet.Trainer(net, optimizer="adam",
        opt_kwargs=dict(learning_rate=0.01))
    path = trainer.train(data_provider, output_path="models", 
        training_iters=100, epochs=100, dropout=0.5, display_step=10,
        restore=True)

    x_test, y_test = get_test_image('data_dir')
    prediction = net.predict(path, x_test)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[1].imshow(np.squeeze(y_test, 0), aspect="auto")
    ax[2].imshow(np.squeeze(prediction, 0), aspect="auto")


if __name__ == '__main__':
    train()
