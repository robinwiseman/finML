import os
import tensorflow as tf


class ModelHandler:
    def __init__(self, chk_path, chk_folder: str = 'model_chkpoints'):
        self.chk_folder = chk_folder
        self.chk_path = self.chkp_path(chk_path)

    def write(self, model, do=False):
        if do:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.write(self.chk_path)

    def restore(self, model, do=False):
        if do:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(self.chk_path)

    def chkp_path(self, path):
        path = os.path.join(self.chk_folder, path)
        return os.path.join(os.path.dirname(__file__), path)