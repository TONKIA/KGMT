"""
Knowledge Graph Multi-Target CDR Model, KGMT
@author: TONKIA
@github: https://github.com/TONKIA
@email: tongxk96@outlook.com
"""

import argparse
from DataSet import CultureDataSet, DoubanDataSet
import numpy as np
import tensorflow as tf
from KGMT import KGMT
from DMF import DMF
import math


class Main:
    def __init__(self, args):
        # 指定模型
        self.model_class = args.model
        # DMF model param
        self.user_layer = args.user_layer
        self.item_layer = args.item_layer
        # KGMT model param
        self.dim = args.dim
        self.reg = args.reg
        self.mlp_dim = args.mlp
        self.unit = args.unit
        # init dataset
        self.dataset_name = args.dataset.lower()
        self.domain = args.domain.lower()
        if self.dataset_name == "culture":
            self.dataset = DoubanDataSet(self.domain, args.test_limit)
        else:
            self.dataset = CultureDataSet(self.domain, args.test_limit)
        self.global_emb = self.dataset.get_global_emb(self.dim)
        # train param
        self.optimizer = tf.optimizers.Adam(args.lr)
        self.epoch = args.epoch
        self.batch = args.batch
        self.early_stop = args.early_stop
        # evaluate parm
        self.k = args.k
        print("=" * 10, "finished loading data", "=" * 10)
        print("dataset:", self.dataset_name)
        print("domain:", self.domain)
        print("dim:", self.dim)
        print("unit:", self.unit)
        print("global embedding shape:", self.global_emb.shape)
        print("train shape:", self.dataset.get_train_data().shape)
        print("test shape:", self.dataset.get_test_data().shape)

    def run(self):
        # init model
        if self.model_class.lower() == "dmf":
            model = DMF(self.user_layer, self.item_layer, self.dataset.get_embedding())
        else:
            model = KGMT(self.dim, self.dataset.n_users, self.dataset.n_items, self.global_emb, unit=self.unit,
                         reg=self.reg, mlp_dim=self.mlp_dim)
        model.build((None, 4))
        model.summary()
        # epoch
        best_hr = -1
        best_ndcg = -1
        best_epoch = -1
        hr_n = [0.] * 10
        ndcg_n = [0.] * 10
        print("=" * 10, "train start", "=" * 10)
        for epo in range(1, self.epoch + 1):
            loss = self.train(model)
            hr, ndcg = self.evaluate(model, self.k)
            for n in range(10):
                hr_n[n] = max(hr_n[n], hr[n])
                ndcg_n[n] = max(ndcg_n[n], ndcg[n])
            print("epoch: {}, loss: {:.3f}, hr@{}: {}, ndcg@{}: {}".format(epo, loss, self.k, hr[self.k - 1], self.k,
                                                                           ndcg[self.k - 1]))
            # print("hr@n:", hr_n)
            # print("ndcg@n:", ndcg_n)
            if hr[self.k - 1] > best_hr:
                best_hr = hr[self.k - 1]
                best_epoch = epo
            if ndcg[self.k - 1] > best_ndcg:
                best_ndcg = ndcg[self.k - 1]
                best_epoch = epo
            if epo - best_epoch > self.early_stop:
                print("early stop")
                break
        print("=" * 10, "train finished", "=" * 10)
        print("dataset:", self.dataset_name)
        print("domain: {}, dim: {}, unit: {}".format(self.domain, self.dim, self.unit))
        print("best epoch: {}, hr@{}: {}, ndcg@{}: {}".format(best_epoch, self.k, best_hr, self.k, best_ndcg))
        print("hr@n:", hr_n)
        print("ndcg@n:", ndcg_n)

    def train(self, model):
        @tf.function
        def loss_func(y, y_predict):
            y_norm = y / self.dataset.max_rating
            loss = y_norm * tf.math.log(y_predict) + (1 - y_norm) * tf.math.log(1 - y_predict)
            loss = -tf.reduce_sum(loss)
            return loss

        @tf.function
        def train_batch(x, y):
            with tf.GradientTape() as tape:
                y_predict = tf.squeeze(model(x, training=True))
                loss = loss_func(y, y_predict) + sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        # train
        losses = []
        train_data = self.dataset.get_train_data()
        num_batch = len(train_data) // self.batch + 1
        for bat in range(num_batch):
            batch_start = self.batch * bat
            batch_end = batch_start + self.batch
            batch_data = train_data[batch_start:batch_end]
            x = batch_data[:, :4]
            y = batch_data[:, 4]
            loss = train_batch(x, y)
            losses.append(loss)
        return np.mean(losses)

    def evaluate(self, model, k):
        test_data = self.dataset.get_test_data()
        hr = [[] for _ in range(k)]
        ndcg = [[] for _ in range(k)]
        for idx in range(len(test_data)):
            test_group = test_data[idx]
            y_predict = model(test_group, training=False).numpy().reshape(-1)
            rank_idx = int(np.where(y_predict.argsort()[::-1] == 0)[0])
            for n in range(k):
                if rank_idx <= n:
                    hr[n].append(1)
                    ndcg[n].append(math.log(2) / math.log(rank_idx + 2))
                else:
                    hr[n].append(0)
                    ndcg[n].append(0)
        return np.mean(hr, axis=1), np.mean(ndcg, axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 选择模型 kgmt or dmf
    parser.add_argument('-model', default='kgmt')
    # 选择数据集 douban or culture
    parser.add_argument('-dataset', default='culture')
    # 选择域 douban(book, movie, music) or culture(book, venue, video)
    parser.add_argument('-domain', default='book')
    # model param DMF
    parser.add_argument('-user_layer', default=[1024, 512, 128, 64])
    parser.add_argument('-item_layer', default=[1024, 512, 128, 64])
    # model param KGMT
    parser.add_argument('-dim', default=32)
    parser.add_argument('-reg', default=0.001)
    parser.add_argument('-mlp', default=[128, 256, 512, 256, 128])
    parser.add_argument('-unit', default=2)
    # train param
    parser.add_argument('-lr', default=0.0001)
    parser.add_argument('-epoch', default=50)
    parser.add_argument('-batch', default=256)
    parser.add_argument('-early_stop', default=5)
    # evaluate param
    parser.add_argument('-k', default=10)
    # test param
    parser.add_argument('-test_limit', default=0)
    # run...
    args = parser.parse_args()
    main = Main(args)
    main.run()
