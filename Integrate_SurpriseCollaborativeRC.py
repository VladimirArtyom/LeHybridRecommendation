import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.accuracy import rmse, mse
from surprise.model_selection import train_test_split
from typing import List , Tuple
np.random.seed(0)


class SurpriseCollaborativeBasedRecommender():
    def __init__(self, datasetPath: str,
                  ratingScale: Tuple,
                  useCols: List[str],
                  measures: List[str] = ["RMSE"]):
        self.df: pd.DataFrame = pd.read_csv(datasetPath, usecols=useCols)
        self.reader: Reader = Reader(rating_scale=ratingScale)
        self.surpriseDF = Dataset.load_from_df(self.df, self.reader)
        self.modelMeasures = measures

        #  Model 
        self.svdModel = SVD()
        self.nmfModel = NMF()
    def prepareRecipeIndices(self, filePath: str, column_name: str):
        df_copy = self.df.copy().reset_index()
        df_copy = df_copy.set_index(column_name)
        self.recipeIndices = pd.Series(df_copy["index"], index=df_copy.index)

    def alignDataset(self, target: List[str]):
        self.df = self.df[target]
        self.items = self.df.iloc[:, 1]
    def splitDataset(self):
        self.train_set, self.test_set = train_test_split(self.surpriseDF, test_size=0.2)
        self.train_set_adam = list(self.train_set.all_ratings())

    def fit(self, w1: float, w2:float ):
        #self.svdModel = SVD(lr_all=w1, biased=True)
        #self.nmfModel = NMF(lr_bu=w2, lr_bi=w2, biased=True)
        self.svdModel = SVD()
        self.nmfModel = NMF()
        self.svdModel.fit(self.train_set)
        self.nmfModel.fit(self.train_set)


    def calculateRMSE(self, squaredErrors: List):
        return np.sqrt(np.mean(squaredErrors))

    def calculateError(self, preds_svd: List, preds_nmf: List,
                        weight_svd: float=0.5,
                        weight_nmf: float=0.5):
        weighted_preds: List = []
        for pred_svd, pred_nmf in zip(preds_svd, preds_nmf):
            squared_error = (pred_svd.est * weight_svd - pred_nmf.est * weight_nmf)**2
            weighted_preds.append(squared_error)
        return weighted_preds
    ## Adam Optimizer Utilities
    def objective_function(self, weights):
        """
        Calculates the objective function value for the given weights.

        Parameters:
            weights (List[float]): The weights used in the calculations.

        Returns:
            float: The root mean squared error (RMSE) of the predictions.
        """
        predictions_svd: List = self.svdModel.test(self.train_set_adam)
        predictions_nmf: List = self.nmfModel.test(self.train_set_adam)
        squaredErrors: List = self.calculateError(predictions_svd, predictions_nmf, weights[0], weights[1] )
        return self.calculateRMSE(squaredErrors)

    def adam_optimizer(self, weights: List[float], 
                       lr: float = 0.1,
                       beta_1: float = 0.9,
                       beta_2: float= 0.999,
                       epsilon: float = 1e-8,
                       h: float = 1e-6,
                       steps: int = 10):
        """
        Performs Adam optimization on the given weights.

        Parameters:
            weights (List[float]): The initial weights for the optimization.
            lr (float, optional): The learning rate. Defaults to 0.1.
            beta_1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
            beta_2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
            epsilon (float, optional): A small constant for numerical stability. Defaults to 1e-8.
            h (float, optional): The step size for calculating gradients. Defaults to 0.01.
            steps (int, optional): The number of optimization steps. Defaults to 50.

        Returns:
            List[float]: The optimized weights.

        Notes:
            - The Adam optimizer is an extension of the Stochastic Gradient Descent (SGD) optimizer that combines both momentum and RMSprop methods.
            - The optimization process involves updating the weights using the gradients calculated from the objective function.
            - The first moment estimate (m1_dw) is updated using the formula: m1_dw = beta_1 * m1_dw + (1 - beta_1) * grads.
            - The second moment estimate (v1_dw) is updated using the formula: v1_dw = beta_2 * v1_dw * (1 - beta_2) * (grads**2).
            - The corrected momentum (m1_dw_corrected) is calculated as: m1_dw_corrected = m1_dw / (1 - beta_1**time_step).
            - The corrected RMSprop (v1_dw_corrected) is calculated as: v1_dw_corrected = v1_dw / (1 - beta_1**time_step).
            - The weights are updated using the formula: weights -= lr * m1_dw_corrected / (np.sqrt(v1_dw_corrected) + epsilon).
            - The optimized weights are printed and returned.
        """
        m1_dw = np.zeros_like(weights)
        v1_dw = np.zeros_like(weights)

        time_step: int = 0 
        for i in range(steps):
            grads: float = np.zeros_like(weights)

            for j in range(len(weights)):
                new_weights = weights
                new_weights[j] += h
                grads[j] = (self.objective_function(new_weights) - self.objective_function(weights)) / h
            
            time_step += 1

            # Update momentum weights
            m1_dw = beta_1 * m1_dw + (1 - beta_1) * grads

            # Update rms weight
            v1_dw = beta_2 * v1_dw * (1 - beta_2) * (grads**2)

            # Corrected momentum
            m1_dw_corrected = m1_dw / (1 - beta_1**time_step)
            v1_dw_corrected = v1_dw / (1 - beta_2**time_step)
            weights = weights -  lr * m1_dw_corrected / (np.sqrt(v1_dw_corrected) + epsilon)

        print("Optimizer Weights", weights)
        return weights

    def predictRecommendation(self, userId, items: List, algo):
        preds: List = []
        preds_iid: List = []
        for item_indx in range(len(items)):
            pred = algo.predict(userId, items[item_indx] )
            if pred.iid not in preds_iid:
                preds_iid.append(pred.iid)
                preds.append(pred)
        preds = sorted(preds, key = lambda x : x.est, reverse=True)
        return preds

    def parallelRecommendation(self, userId: int,
                                totalParalel: int):
        
        low = 0.01
        high = 1.00
        results = {}
        startingWeights = [[np.random.uniform(low, high), np.random.uniform(low, high)] for i in range(totalParalel)]
        for indx_weight in range(len(startingWeights)):
            w1, w2 = self.adam_optimizer(startingWeights[indx_weight])
            predsSvd = self.predictRecommendation(userId, self.items, self.svdModel)
            predsNMF = self.predictRecommendation(userId, self.items, self.nmfModel)
            rmse_value = self.calculateRMSE(self.calculateError(predsSvd, predsNMF, w1, w2))
            results[f"res_{indx_weight}"] = {
                "w1": w1,
                "w2": w2,
                "rmse": rmse_value,
                "preds_svd": predsSvd,
                "preds_nmf": predsNMF
            }
        return sorted(results.items(), key=lambda x: x[1]["rmse"])
        
    def getRecommendations(self, userId, topN: int, totalParallel: int = 2 ):
        results = self.parallelRecommendation(userId, totalParallel)[0]
        w1 = topN *(1- results[1]["w1"] )
        w2 = topN *(1- results[1]["w2"] ) 
        print(w1, w2)
        takeSvd: int = int(np.ceil(w1))
        takeNMF: int = int(np.ceil(w2))
        print(f"takeSvd: {takeSvd}, takeNMF: {takeNMF}")
        predsSVDTop = results[1]["preds_svd"][:takeSvd]
        predsNMFTop = results[1]["preds_nmf"][:takeNMF]
        
        return predsSVDTop, predsNMFTop

"""
if __name__ == "__main__":
    datasetPathReviews: str = "reviews_filtered.csv"
    useCols: List[str] = ["RecipeId", "AuthorId", "Rating"]
    targetAlign: List[str] = ["AuthorId", "RecipeId", "Rating"]
    ratingScale: Tuple = (1, 5)
    measures: List[str] = ["RMSE"]
    weights: List[float] = [0.1, 0.1]

    surpriseCF = SurpriseCollaborativeBasedRecommender(datasetPathReviews, ratingScale,
                                                       useCols, measures)
    surpriseCF.alignDataset(targetAlign)

    surpriseCF.adam_optimizer(weights)
"""