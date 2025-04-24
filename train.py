# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # 用於繪製 PSNR 曲線圖

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil
import dnnlib.util as util

import config

from util import save_image, save_snapshot
from validation import ValidationSet
from dataset import create_dataset

class AugmentGaussian:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x):
        (minval, maxval) = self.train_stddev_range
        shape = tf.shape(x)
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
        return x + tf.random_normal(shape) * rng_stddev

    def add_validation_noise_np(self, x):
        return x + np.random.normal(size=x.shape)*(self.validation_stddev/255.0)

class AugmentPoisson:
    def __init__(self, lam_max):
        self.lam_max = lam_max

    def add_train_noise_tf(self, x):
        chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=self.lam_max)
        return tf.random_poisson(chi_rng*(x+0.5), shape=[])/chi_rng - 0.5

    def add_validation_noise_np(self, x):
        chi = 30.0
        return np.random.poisson(chi*(x+0.5))/chi - 0.5

def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5 + np.cos(t * np.pi) / 2)**2
        return learning_rate * smooth
    return learning_rate

def train(
    submit_config: dnnlib.SubmitConfig,
    iteration_count: int,
    eval_interval: int,
    minibatch_size: int,
    learning_rate: float,
    ramp_down_perc: float,
    noise: dict,
    validation_config: dict,
    train_tfrecords: str,
    noise2noise: bool):

    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    ctx = dnnlib.RunContext(submit_config, config)
    tfutil.init_tf(config.tf_config)

    dataset_iter = create_dataset(train_tfrecords, minibatch_size, noise_augmenter.add_train_noise_tf)

    with tf.device("/gpu:0"):
        net = tflib.Network(**config.net_config)

    net.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device("/cpu:0"):
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])

        noisy_input, noisy_target, clean_target = dataset_iter.get_next()
        noisy_input_split = tf.split(noisy_input, submit_config.num_gpus)
        noisy_target_split = tf.split(noisy_target, submit_config.num_gpus)
        clean_target_split = tf.split(clean_target, submit_config.num_gpus)

    opt = tflib.Optimizer(learning_rate=lrate_in, **config.optimizer_config)

    for gpu in range(submit_config.num_gpus):
        with tf.device("/gpu:%d" % gpu):
            net_gpu = net if gpu == 0 else net.clone()

            denoised = net_gpu.get_output_for(noisy_input_split[gpu])

            if noise2noise:
                meansq_error = tf.reduce_mean(tf.square(noisy_target_split[gpu] - denoised))
            else:
                meansq_error = tf.reduce_mean(tf.square(clean_target_split[gpu] - denoised))
            with tf.control_dependencies([autosummary("Loss", meansq_error)]):
                opt.register_gradients(meansq_error, net_gpu.trainables)

    train_step = opt.apply_updates()

    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    summary_log.add_graph(tf.get_default_graph())

    psnr_values = []

    print('Training...')
    time_maintenance = ctx.get_time_since_last_update()
    ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=0, max_epoch=iteration_count)

    for i in range(iteration_count):
        if ctx.should_stop():
            break

        if i % eval_interval == 0:
            time_train = ctx.get_time_since_last_update()
            time_total = ctx.get_time_since_start()

            [source_mb, target_mb] = tfutil.run([noisy_input, clean_target])
            denoised = net.run(source_mb)

            mse = np.mean((denoised - target_mb) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

            print(f'iter {i}, PSNR: {psnr:.2f} dB')

            save_image(submit_config, denoised[0], f"img_{i}_y_pred.png")
            save_image(submit_config, target_mb[0], f"img_{i}_y.png")
            save_image(submit_config, source_mb[0], f"img_{i}_x_aug.png")

            validation_set.evaluate(net, i, noise_augmenter.add_validation_noise_np)

            dnnlib.tflib.autosummary.save_summaries(summary_log, i)
            ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=i, max_epoch=iteration_count)
            time_maintenance = ctx.get_last_update_interval() - time_train

        lrate = compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        tfutil.run([train_step], {lrate_in: lrate})

    plt.figure()
    plt.plot(range(0, iteration_count, eval_interval), psnr_values, label='PSNR')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Curve')
    plt.legend()
    plt.grid()
    plt.savefig(submit_config.run_dir + '/plot_psnr.png')
    plt.close()

    print("Elapsed time: {0}".format(util.format_time(ctx.get_time_since_start())))
    save_snapshot(submit_config, net, 'final')

    summary_log.close()
    ctx.close()
