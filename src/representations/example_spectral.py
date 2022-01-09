import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Any
import tensorflow as tf

from utils.utils import ModelHandler
from representations.datagen import DataGenCheckerBoard
from representations.spectral_models import (
    GraphLaplacian,
    SpectralCFClusterModel
)


def plot_gl(gl, dg):
    k, eigs, eigs_lower = gl.cluster_number()
    ei = np.arange(0, len(eigs))
    eil = np.arange(0, k)
    df1 = pd.DataFrame({'eigenvalues': eigs, 'eigenvalue_index': ei})
    df2 = pd.DataFrame({'lower eigenvalues': eigs_lower, 'eigenvalue_index': eil})
    font_size = 16
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('figure', titlesize=font_size)
    ax1 = df1.plot(kind='scatter', x='eigenvalue_index',
                   y='eigenvalues', label='eigenvalues',
                   figsize=(10, 6))
    ax1 = df2.plot(kind='scatter', x='eigenvalue_index',
                   y='lower eigenvalues', color='r', label='lower eigenvalues',
                   figsize=(10, 6), ax=ax1)
    ax1.set_xlabel("eigenvalue index")
    ax1.set_ylabel("eigenvalue")
    ax1.set_title(f"Graph Laplacian Eigenvalues: cluster periodicities {dg.cluster_periodicities}")
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(8, 6))
        ax = sns.heatmap(dg.R[:20, :20], square=True)

    plt.show()


def plot_post_train(model):
    users = np.arange(0, model.n_users)
    items = np.arange(0, model.n_items)
    items_ = np.array([x + model.n_users for x in items])
    u_emb, i_emb, _ = model.batch_embeddings(users,
                                             items_,
                                             items_)
    R_post = np.zeros(shape=(model.n_users, model.n_items))
    for i in range(model.n_users):
        for j in range(model.n_items):
            R_post[i, j] = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(u_emb[i], i_emb[j])))

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(8, 6))
        ax = sns.heatmap(R_post[:20, :20], square=True)


def plot_post_train_3D(model):
    # special plot case for cluster_periodicities = [3]
    users = np.arange(0, model.n_users)
    items = np.arange(0, model.n_items)
    items_ = np.array([x + model.n_users for x in items])
    u_emb, i_emb, _ = model.raw_embeddings(users,
                                           items_,
                                           items_)
    font_size = 16
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('figure', titlesize=font_size)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    markers = ['x', '^', '+']
    mkrs = {}
    for i in range(model.n_users):
        mkr = markers[i % 3]
        mkr_vals = mkrs.get(mkr, {'x': [], 'y': [], 'z': []})
        mkr_vals['x'].append(u_emb[i, 0])
        mkr_vals['y'].append(u_emb[i, 1])
        mkr_vals['z'].append(u_emb[i, 2])
        mkrs[mkr] = mkr_vals

    counter = 0
    labels = ['C1', 'C2', 'C3']
    for mkr, mkr_vals in mkrs.items():
        ax.scatter(mkr_vals['x'], mkr_vals['y'], mkr_vals['z'], marker=mkr,
                   label=labels[counter], s=300)
        counter += 1

    ax.set_title(f"Cluster vectors : checkerboard periodicity: [3]")


def main(chk_path:str, train=True, restore=False, examine_gl=False, run_eager=False):
    tf.config.run_functions_eagerly(run_eager)
    mh = ModelHandler(chk_path=chk_path)
    noise_scale = 0.0
    batch_size = 500
    n_users = 50
    n_items = 50
    cluster_periodicities = [3] # control the number of clusters in the dummy data

    dg = DataGenCheckerBoard(batch_size=batch_size,
                          n_users=n_users,
                          n_items=n_items,
                          cluster_periodicities=cluster_periodicities,
                          noise_scale=noise_scale)
    gl = GraphLaplacian(dg.R)
    if examine_gl:
        plot_gl(gl, dg)

    latent_vector_size, _, _ = gl.cluster_number()
    loss_regularization_weight = 1e-5
    model = SpectralCFClusterModel(dg.R,
                                   dg.n_users,
                                   dg.n_items,
                                   latent_vector_size=latent_vector_size,
                                   loss_regularization_weight=loss_regularization_weight)

    n_epochs = 100
    n_steps = 20
    best_loss = 1e9
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    if train:
        mh.restore(model=model, do=restore)  # optionally restore model from earlier run
        for epoch in range(1, n_epochs + 1):
            users, pos_items, neg_items = dg.sample()
            # the [user, item] is a bipartite matrix generated from separately numbered lists of items
            # offset the item indices to enable lookup into the bipartite matrix (and associated vector embeddings
            # for items maintained by the model
            pos_items = [x+model.n_users for x in pos_items] # bipartite matrix in the model - adjust raw data indices for upper right block
            neg_items = [x+model.n_users for x in neg_items] # bipartite matrix in the model - adjust raw data indices for upper right block
            for step in range(n_steps):
                with tf.GradientTape() as tape:
                    model()
                    loss = model.loss(users, pos_items, neg_items)
                    if (step == n_steps-1) or ((step == 0) and (epoch == 1)):
                        print(f"""Epoch: {epoch}/{n_epochs} : Step {step} : Loss: {np.round(loss,5)} : """, end='')
                        save_condition = (not np.isnan(loss)) and (loss < best_loss)

                        if save_condition:
                            print(": saving : ")
                            mh.write(model=model, do=True)
                            best_loss = loss
                        else:
                            print("")

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    mh.restore(model=model, do=restore)  # optionally restore model from earlier run

    return model, dg


if __name__=="__main__":
    model_type = 'spectral_cluster'
    run = 0
    chk_path = f'chk_{model_type}_{run}'
    main(chk_path=chk_path, train=True, restore=False, run_eager=True)