import tensorflow as tf
import numpy as np


class SpectralEmbeddingLayer(tf.Module):
    def __init__(self, n_users,
                 n_items,
                 latent_vector_size_in: int = 2,
                 initializer_mean: np.float32 = 0.01,
                 initializer_sdev: np.float32 = 0.02):
        super(SpectralEmbeddingLayer, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.latent_vector_size_in = latent_vector_size_in
        self.initializer_mean = initializer_mean
        self.initializer_sdev = initializer_sdev

        initializer = tf.random_normal_initializer(mean=self.initializer_mean,
                                                   stddev=self.initializer_sdev)

        embedding_shape = [n_users+n_items, latent_vector_size_in]
        self.embeddings = tf.Variable(initial_value=initializer(shape=embedding_shape, dtype=tf.float32),
                                        name=f"embeddings_{0}",
                                        trainable=True)

    @tf.function
    def __call__(self):
        return self.embeddings


class BPRLoss(tf.keras.losses.Loss):
    def __init__(self, reg_weight: np.float32 = 0.01):
        super(BPRLoss, self).__init__()
        self.reg_weight = reg_weight

    def __call__(self, u_embeddings, pos_i_embeddings, neg_i_embeddings):
        '''
        Use Bayesian Personalised Ranking loss on (user, positive item, negative item) triplets
        The minimization of the loss function seeks to maximize the
        probability that the model thinks the user likes the positive item more than the negative item.
        '''
        batch_size = u_embeddings.shape[0]+pos_i_embeddings.shape[0]+neg_i_embeddings.shape[0]
        pos_scores = tf.reduce_sum(tf.multiply(u_embeddings, pos_i_embeddings), 1)
        neg_scores = tf.reduce_sum(tf.multiply(u_embeddings, neg_i_embeddings), 1)
        regularizer = tf.nn.l2_loss(u_embeddings) + tf.nn.l2_loss(pos_i_embeddings) + tf.nn.l2_loss(neg_i_embeddings)
        regularizer = regularizer / batch_size

        Lp = tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_mean(Lp)) + self.reg_weight * regularizer
        return loss


class GraphLaplacian():
    def __init__(self, graph: np.array):
        '''
        :param graph: upper right block of a bipartite graph of dimension [n_u, n_i], represented as an
        adjacency matrix
        '''
        self.graph = graph
        self.N_u = graph.shape[0]
        self.N_i = graph.shape[1]
        self.N = self.N_u+self.N_i
        self.A = self.adjacency_matrix()
        self.D = self.degree_matrix()
        self.L = self.laplacian_matrix()
        self.svd_u, self.lambda_, self.svd_vh = np.linalg.svd(self.L)
        # order with eigenvalues in ascending value (svd gives them in descending)
        # spectral clustering uses the lowest-k eigenvalues + associated eigenvectors
        self.lambda_ = np.diag(np.flip(self.lambda_))
        self.svd_vh = np.flip(self.svd_vh, axis=0)
        self.svd_u = np.flip(self.svd_u, axis=1)

    def adjacency_matrix(self):
        A = np.zeros([self.N, self.N], dtype=np.float32)
        # bipartite graph matrix : off diagonal blocks non-zero
        A[:self.N_u, self.N_u:] = self.graph
        A[self.N_u:, :self.N_u] = self.graph.T
        return A

    def degree_matrix(self):
        # a measure of how much each node is connected to other nodes
        degree = np.sum(self.A, axis=1, keepdims=False)
        return degree

    def laplacian_matrix(self):
        # Note: this is the symmetric form 1 - D^(-1/2)*AD^(-1/2) of the graph laplacian.
        Dinv = np.diag(np.power(self.D, -0.5))
        DinvA = np.dot(np.dot(Dinv, self.A), Dinv)
        return np.identity(self.N, dtype=np.float32) - DinvA

    def cluster_number(self):
        # a heuristic for selecting the number of clusters
        eigs = np.diagonal(self.lambda_)
        eig_q = np.quantile(eigs, 0.5)
        eigs_lower = eigs[eigs < eig_q * 0.98]
        return len(eigs_lower), eigs, eigs_lower


class SpectralCFClusterModel(tf.Module):
    def __init__(self, graph, n_users, n_items, latent_vector_size: int=2,
                 loss_regularization_weight=0.01):
        super(SpectralCFClusterModel, self).__init__()
        self.gl = GraphLaplacian(graph)
        self.graph = graph
        self.n_users = n_users
        self.n_items = n_items
        self.latent_vector_size = latent_vector_size # define the size of cluster space

        self.loss_fn = BPRLoss(reg_weight=loss_regularization_weight)
        self.spectral_projection = self.get_spectral_projection()
        vars(self)[f'projection_{0}'] = SpectralEmbeddingLayer(n_users,
                                                    n_items,
                                                    latent_vector_size_in=latent_vector_size)

    def __call__(self):
        # Note: no inputs in this model: the implicit vectors are what are learned.
        # No explicit features passed in
        x = self.projection_0() # first layer has no inputs, it just uses its embeddings
        # the input embeddings in the layer are learnable
        # Note: the embeddings used to fit the bipartite graph matrix are the results of the projection returned
        # from the layer i.e. projected into "cluster" space - the truncated rows of the eigenvector matrix of the graph
        # Laplacian
        self.embeddings = x

    def get_spectral_projection(self):
        u_truncated = self.gl.svd_u[:,0:self.latent_vector_size]
        norm = np.sqrt(1 / np.sum(np.multiply(u_truncated, u_truncated), axis=1, keepdims=True))
        u_truncated = np.multiply(u_truncated, norm)
        u_trun = tf.Variable(initial_value=u_truncated, name=f"u_truncated",trainable=False)
        return u_trun

    def loss(self, users, pos_items, neg_items):
        u_embeddings, pos_i_embeddings, neg_i_embeddings = self.batch_embeddings(users, pos_items, neg_items)
        return self.loss_fn(u_embeddings, pos_i_embeddings, neg_i_embeddings)

    def batch_embeddings(self, u_batch_idx, pos_i_batch_idx, neg_i_batch_idx):
        '''
        :return: Return batch embeddings projected into spectral cluster space
        '''
        self.u_batch_embeddings = self.project_embeddings(tf.nn.embedding_lookup(self.embeddings, u_batch_idx))
        self.pos_i_batch_embeddings = self.project_embeddings(tf.nn.embedding_lookup(self.embeddings, pos_i_batch_idx))
        self.neg_i_batch_embeddings = self.project_embeddings(tf.nn.embedding_lookup(self.embeddings, neg_i_batch_idx))

        return self.u_batch_embeddings, self.pos_i_batch_embeddings, self.neg_i_batch_embeddings

    def raw_embeddings(self, u_batch_idx, pos_i_batch_idx, neg_i_batch_idx):
        # NOTE: this function isn't used on the graph during training - just a
        # convenience to access the raw embeddings (before projection)
        raw_u_batch_embeddings = np.array(tf.nn.embedding_lookup(self.embeddings, u_batch_idx))
        raw_pos_i_batch_embeddings = np.array(tf.nn.embedding_lookup(self.embeddings, pos_i_batch_idx))
        raw_neg_i_batch_embeddings = np.array(tf.nn.embedding_lookup(self.embeddings, neg_i_batch_idx))

        return raw_u_batch_embeddings, raw_pos_i_batch_embeddings, raw_neg_i_batch_embeddings

    @tf.function
    def project_embeddings(self, embeddings):
        return tf.transpose(tf.matmul(self.spectral_projection, tf.transpose(embeddings)))

