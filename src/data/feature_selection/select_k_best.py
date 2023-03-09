from transformer import Transformer
from src.stats import f_classification, f_regression
from data.dataset import Dataset

class SelectKBest(Transformer):

    def __init__(self, k: int, score_func=f_classification):
        """The SelectKBest method selects the features according to the k highest scores
        computed using a scoring function. 
        :param k: Number of feature with best score to be selected
        :type k: int
        :param score_func: The scoring function, defaults to f_classif
        :type score_func: callable, optional
        -------------------------------------------------------------------------
        In this implementation we will consider the two F-statistics functions, 
        one for regression (f_regress) and the other for classification tasks (f_classif).
        The p and F values have an inverse relationship, the greater
        the F value the lesser the p.
        Larger values of F correspond to a rejection with probability
        (1-p) of the null hypothesis, meaning that the corresponding 
        features has an effect on the predictions.
        """
        self.k = k
        self.score_func = score_func

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)

    def transform(self, dataset, inline=False) -> Dataset:
        # documentar
        if not dataset.all_numeric:
            raise ValueError("Theres is not encoded data. Consider using an encoder.")
        else:
            top_k_indices = self.F.argsort()[-self.k:][::-1]
        if inline:
            dataset.X = dataset.X[:, top_k_indices]
        else:
            return Dataset(dataset.X[:, top_k_indices], dataset.y, feature_names=dataset.feature_names[top_k_indices])
        pass
