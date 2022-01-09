import numpy as np


class DataGenCheckerBoard(object):
    def __init__(self, batch_size: int = 10, n_users: int = 20, n_items: int=20,
                 cluster_periodicities=[2, 3], noise_scale = 0.0, symmetry_break=False):
        '''
        Highly structured dummy data with a defined set of clusters of
        users, items and which users like which items.
        :param batch_size:
        :param n_users:
        :param n_items:
        :param observation_noise:
        :param cluster_num: drives the number of clusters
        '''
        self.batch_size = batch_size
        self.n_users = n_users
        self.n_items = n_items
        self.noise_scale = noise_scale
        self.cluster_periodicities = cluster_periodicities
        self.cluster_weights = [1/x for x in self.cluster_periodicities]
        self.R = None
        self.R_full = None
        self.R_unobserved = None
        self.symmetry_break = symmetry_break
        self.generate_implicit_feedback()

    def generate_implicit_feedback(self):
        # highly structured dummy data
        # create a multicoloured checkerboard of implicit feedback by summing
        # checkerboards of different periodicities
        R = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for periodicity, weight in zip(self.cluster_periodicities, self.cluster_weights):
            R += self.generate_checkerboard(periodicity)*weight

        if self.symmetry_break:
            R = self.apply_symmetry_break(R)

        self.R_full = R.copy()
        # set some of the observations to zero at random
        if self.noise_scale > 0.0:
            print("applying missing at random")
            mar = np.argwhere(np.random.binomial(n=1, p=self.noise_scale, size=(self.n_users, self.n_items))>0.0)
            if len(mar):
                R[tuple(np.transpose(mar))] = 0.0

        # retain "unobserved" interest observations
        self.R_unobserved = self.R_full - R

        self.R = R

    def generate_checkerboard(self, periodicity):
        R = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for i in range(self.n_users):
            for j in range(self.n_items):
                if ((i+j) % periodicity) == 0:
                    R[i,j] = 1

        return R

    def apply_symmetry_break(self, r: np.array, symmetry_break: float =1e-4):
        # enable u-i pairs with otherwise identical profile in the same cluster to meet
        # during sampling
        r += np.random.uniform(0, symmetry_break, size=(self.n_users, self.n_items))
        return r

    def sample(self, num_per_user_sample: int = 10):
        user_sample_size = int(self.batch_size/num_per_user_sample)
        users = np.random.choice(range(self.n_users), user_sample_size, replace=True)
        # for each user sample, pick a single positive item, and a negative item (relative to the
        # positive item for that user)
        pos_items = np.array([], dtype=np.int32)
        neg_items = np.array([], dtype=np.int32)
        final_users = np.array([], dtype=np.int32)
        for u in users:
            for _ in range(user_sample_size): # multiple samples per user
                # Random sample positive items from all items where the R matrix is greater than
                # zero. Then pick a corresponding negative pair for the user relative to the
                # positive item.
                # Handle the case that a user has all elements >0, and the positive is picked
                # as the smallest, by picking again until a larger positive is picked
                # Assume that the pathological case that a user has all elements > 0 and equal
                # does not exist (given we are constructing the data set, and can explicitly
                # design it not to)
                relative_negs = np.array([])
                neg_item = None
                neg_tries = 0
                max_neg_tries = 10
                while not len(relative_negs):
                    if neg_tries > max_neg_tries:
                        break
                    pos_item = np.random.choice(np.where(self.R[u, :] > 0)[0])
                    relative_negs = np.where(self.R[u, :] < self.R[u, pos_item])[0]
                    if len(relative_negs):
                        neg_item = np.random.choice(relative_negs)
                    neg_tries += 1

                if neg_item is not None:
                    pos_items = np.append(pos_items, pos_item)
                    neg_items = np.append(neg_items, neg_item)
                    final_users = np.append(final_users, u)

        return final_users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items
