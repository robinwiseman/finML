import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class DataGenBinaryClassifierScores:
    def __init__(self, class_params: Dict[Any, Any], c0_sample_size: int =200,
                 c1_sample_size: int =200):
        '''
        :param class_params: alpha, beta (beta distribution parameters) for each class
        :param c0_sample_size: the sample can be imbalanced in c0 and c1 classes
        :param c1_sample_size:
        '''
        self.class0_alpha = class_params.get('class0_alpha', 2)
        self.class0_beta = class_params.get('class0_beta', 2)
        self.class1_alpha = class_params.get('class1_alpha', 1)
        self.class1_beta = class_params.get('class1_beta', 1)
        self.c0_sample_size = c0_sample_size
        self.c1_sample_size = c1_sample_size
        self.scores = None

    def generate_samples(self):
        class_0 = np.random.beta(self.class0_alpha, self.class0_beta, size=self.c0_sample_size)
        class_1 = np.random.beta(self.class1_alpha, self.class1_beta, size=self.c1_sample_size)
        self.scores = {'class_0': class_0, 'class_1': class_1}
        return self.scores

    @staticmethod
    def plot(scores: Optional[Dict[str, np.array]] = None, set_scale=True):
        dfs0 = pd.DataFrame({'score': scores['class_0']})
        dfs0['class'] = 'c0'
        dfs1 = pd.DataFrame({'score': scores['class_1']})
        dfs1['class'] = 'c1'
        df = pd.concat([dfs0, dfs1])
        if set_scale:
            sns.set_theme(font_scale=1.3)
        g = sns.FacetGrid(df, col="class", height=3.5, aspect=1.1)
        for i in range(2):
            g.axes[0, i].set_xlabel('score', fontsize=16)
            g.axes[0, i].set_ylabel('count', fontsize=16)
            g.axes[0, i].title._text = f"""class = {i}"""
            g.axes[0, i]._left_title.text = f"""class = {i}"""
            g.axes[0, i]._right_title.text = f"""class = {i}"""
        g.map(sns.histplot, "score")
        plt.show()
