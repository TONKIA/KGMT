"""
Knowledge Graph Multi-Target CDR Model, KGMT
@author: TONKIA
@github: https://github.com/TONKIA
@email: tongxk96@outlook.com
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import random


class CultureDataSet:
    def __init__(self, domain, test_limit):
        self.domain = domain
        self.test_limit = test_limit
        file_path = "./data/culture/out/" + domain + ".csv"
        print("read file:", file_path)
        self.data = pd.read_csv(file_path)
        self.users = set(self.data["uid"].values)
        self.items = set(self.data["vid"].values)
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.count = len(self.data)
        self.max_rating = 1.
        self.uid2guid = {}
        self.vid2gid = {}
        self.train_data_pos = []
        self.test_data = defaultdict(list)
        self.user_ranked_item = defaultdict(set)
        # init test data and other properties
        self.init_test_data()
        self.sparsity = 100 * self.count / self.n_users / self.n_items
        print("dataset:", domain)
        print("users: {}, items: {}, max_rating: {}, sparsity: {:.2f}%".format(self.n_users, self.n_items,
                                                                               self.max_rating, self.sparsity))

    def init_test_data(self):
        self.data.sort_values(by=['uid'], axis=0, ascending=[True], inplace=True)
        for _, row in self.data.iterrows():
            uid = int(row["uid"])
            guid = int(row["guid"])
            vid = int(row["vid"])
            gid = int(row["gid"])
            self.uid2guid[uid] = guid
            self.vid2gid[vid] = gid
            if len(self.user_ranked_item[uid]) == 0:
                self.test_data[uid].append([uid, vid, guid, gid])
            else:
                self.train_data_pos.append((uid, vid, guid, gid, 1.))
            self.user_ranked_item[uid].add(vid)

        for uid in self.test_data.keys():
            candidate = self.items - self.user_ranked_item[uid]
            nag_sample = random.sample(candidate, 99)
            for vid in nag_sample:
                self.user_ranked_item[uid].add(vid)
                self.test_data[uid].append([uid, vid, self.uid2guid[uid], self.vid2gid[vid]])

    def get_train_data(self):
        train_data_neg = []
        for uid in self.users:
            candidate = self.items - self.user_ranked_item[uid]
            sample_num = min(len(candidate), len(self.user_ranked_item[uid]) - 100)
            nag_sample = random.sample(candidate, sample_num)
            for vid in nag_sample:
                train_data_neg.append([uid, vid, self.uid2guid[uid], self.vid2gid[vid], .0])
        train_epoch = self.train_data_pos + train_data_neg
        random.shuffle(train_epoch)
        return np.array(train_epoch, dtype=np.float32)

    def get_test_data(self):
        test_data_limited = []
        for uid in self.users:
            user_train_pos_count = len(self.user_ranked_item[uid]) - 100
            if user_train_pos_count > self.test_limit:
                test_data_limited.append(self.test_data[uid])
        return np.array(test_data_limited, dtype=np.float32)

    def get_embedding(self):
        matrix = np.zeros([self.n_users, self.n_items], dtype=np.float32)
        for user, item, _, _, rating in self.train_data_pos:
            matrix[user][item] = rating
        return np.array(matrix)

    def get_global_emb(self, dim):
        return np.load('./data/culture/emb/graph-' + str(dim) + '.npy')


class DoubanDataSet:
    def __init__(self, domain, test_limit):
        self.domain = domain
        self.test_limit = test_limit
        file_path = "./data/douban/out/" + domain + "_rating.csv"
        print("read file:", file_path)
        self.data = pd.read_csv(file_path)
        self.users = set(self.data["uid"].values)
        self.items = set(self.data["vid"].values)
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.count = len(self.data)
        self.max_rating = 0
        self.uid2guid = {}
        self.vid2gid = {}
        self.train_data_pos = []
        self.test_data = defaultdict(list)
        self.user_ranked_item = defaultdict(set)
        # init test data and other properties
        self.init_test_data()
        self.sparsity = 100 * self.count / self.n_users / self.n_items
        print("dataset:", domain)
        print("users: {}, items: {}, max_rating: {}, sparsity: {:.2f}%".format(self.n_users, self.n_items,
                                                                               self.max_rating, self.sparsity))

    def init_test_data(self):
        self.data.sort_values(by=['uid'], axis=0, ascending=[True], inplace=True)
        for _, row in self.data.iterrows():
            uid = int(row["uid"])
            guid = int(row["guid"])
            vid = int(row["vid"])
            gid = int(row["gid"])
            rating = float(row["rating"])
            self.uid2guid[uid] = guid
            self.vid2gid[vid] = gid
            self.max_rating = max(self.max_rating, rating)
            if len(self.user_ranked_item[uid]) == 0:
                self.test_data[uid].append([uid, vid, guid, gid])
            else:
                self.train_data_pos.append((uid, vid, guid, gid, rating))
            self.user_ranked_item[uid].add(vid)

        for uid in self.test_data.keys():
            candidate = self.items - self.user_ranked_item[uid]
            nag_sample = random.sample(candidate, 99)
            for vid in nag_sample:
                self.user_ranked_item[uid].add(vid)
                self.test_data[uid].append([uid, vid, self.uid2guid[uid], self.vid2gid[vid]])

    def get_train_data(self):
        train_data_neg = []
        for uid in self.users:
            candidate = self.items - self.user_ranked_item[uid]
            sample_num = min(len(candidate), len(self.user_ranked_item[uid]) - 100)
            nag_sample = random.sample(candidate, sample_num)
            for vid in nag_sample:
                train_data_neg.append([uid, vid, self.uid2guid[uid], self.vid2gid[vid], .0])
        train_epoch = self.train_data_pos + train_data_neg
        random.shuffle(train_epoch)
        return np.array(train_epoch, dtype=np.float32)

    def get_test_data(self):
        test_data_limited = []
        for uid in self.users:
            user_train_pos_count = len(self.user_ranked_item[uid]) - 100
            if user_train_pos_count > self.test_limit:
                test_data_limited.append(self.test_data[uid])
        return np.array(test_data_limited, dtype=np.float32)

    def get_embedding(self):
        matrix = np.zeros([self.n_users, self.n_items], dtype=np.float32)
        for user, item, _, _, rating in self.train_data_pos:
            matrix[user][item] = rating
        return np.array(matrix)

    def get_global_emb(self, dim):
        return np.load('./data/douban/emb/graph-' + str(dim) + '.npy')
