import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from surprise import Dataset, Reader
from surprise import SVD, NMF, KNNBasic
from surprise.accuracy import rmse, mse
from surprise.model_selection import train_test_split
from typing import List , Tuple
np.random.seed(0)


class SurpriseCollaborativeBasedRecommender():
    def __init__(self, datasetPath: str,
                  ratingScale: Tuple,
                  useCols: List[str]):
        """
        Initializes a SurpriseCollaborativeBasedRecommender object.

        Args:
            datasetPath (str): The path to the dataset file.
            ratingScale (Tuple): The rating scale used in the dataset.
            useCols (List[str]): The columns to be used from the dataset.

        Initializes the SurpriseCollaborativeBasedRecommender object with the given dataset and rating scale.
        Reads the dataset from the specified path using the specified columns and creates a Surprise dataset object.
        Initializes the SVD and NMF models used for collaborative filtering.

        Returns:
            None
        """
        self.df: pd.DataFrame = pd.read_csv(datasetPath, usecols=useCols)
        self.reader: Reader = Reader(rating_scale=ratingScale)
        self.surpriseDF = Dataset.load_from_df(self.df, self.reader)
        #  Model 
        self.svdModel = SVD(lr_all=0.005, reg_all=0.02, n_epochs=50, n_factors=20,
                            biased=True)
        self.nmfModel = NMF(n_factors=200, reg_pu=0.1, reg_bu=0.1,
                            reg_bi=0.1, lr_bu=0.002, lr_bi=0.005 , biased=True)
    def prepareRecipeIndices(self, column_name: str):
        """
        Prepare the recipe indices for the given column name.

        Args:
            column_name (str): The name of the column to set as the index.

        Returns:
            None
        """
        df_copy = self.df.copy().reset_index()
        df_copy = df_copy.set_index(column_name)
        self.recipeIndices = pd.Series(df_copy["index"], index=df_copy.index)

    def alignDataset(self, target: List[str]):
        """
        Aligns the dataset based on the given target column names.

        Args:
            target (List[str]): A list of column names to use as the target columns.

        Returns:
            None

        This function takes in a list of column names as input and uses them to subset the dataframe. It sets the subsetted dataframe as the new dataframe and updates the items attribute with the values from the second column of the subsetted dataframe.
        """
        self.df = self.df[target]
        self.items = self.df.iloc[:, 1]

    def splitDataset(self):
        """
        Splits the dataset into training and testing sets using the train_test_split function from the scikit-learn library.

        Parameters:
            None

        Returns:
            None

        This function splits the dataset into training and testing sets using the train_test_split function from the scikit-learn library. The training set is assigned to the 'train_set' attribute of the object, and the testing set is assigned to the 'test_set' attribute of the object. The 'train_set' is converted to a list of ratings and assigned to the 'train_set_adam' attribute of the object.
        """
        self.train_set, self.test_set = train_test_split(self.surpriseDF, test_size=0.2)
        self.train_set_adam = list(self.train_set.all_ratings())

    def fit(self, w1: float, w2:float ):
        """
        Fits the SVD and NMF models using the given weights.

        Parameters:
            w1 (float): The weight for the SVD model.
            w2 (float): The weight for the NMF model.

        Returns:
            None

        This function fits the SVD and NMF models using the training set. The weights w1 and w2 are used to configure the models. The SVD model is fitted using the default parameters, but the NMF model is fitted with the specified weights.

        Note:
            - The SVD model is fitted using the default parameters.
            - The NMF model is fitted with the specified weights.

        """
        #self.svdModel = SVD(lr_all=w1, biased=True)
        #self.nmfModel = NMF(lr_bu=w2, lr_bi=w2, biased=True)
        #self.svdModel = SVD(lr_all=0.005, reg_all=0.02, n_epochs=50, n_factors=20,
        #                    biased=True)
        #self.nmfModel = NMF(n_factors=200, reg_pu=0.1, reg_bu=0.1,
        #                    reg_bi=0.1, lr_bu=0.002, lr_bi=0.005 , biased=True)
        self.svdModel.fit(self.train_set)
        self.nmfModel.fit(self.train_set)


    def calculateRMSE(self, squaredErrors: List):
        """
        Calculate the root mean squared error (RMSE) from a list of squared errors.

        Parameters:
            squaredErrors (List): A list of squared errors.

        Returns:
            float: The calculated RMSE value.

        """
        return np.sqrt(np.mean(squaredErrors))

    def calculateError(self, preds_svd: List, preds_nmf: List,
                        weight_svd: float=0.5,
                        weight_nmf: float=0.5):
        """
        Calculate the weighted squared errors between predicted values from the SVD and NMF models.

        Parameters:
            preds_svd (List): A list of predicted values from the SVD model.
            preds_nmf (List): A list of predicted values from the NMF model.
            weight_svd (float, optional): The weight to assign to the SVD model's predictions. Defaults to 0.5.
            weight_nmf (float, optional): The weight to assign to the NMF model's predictions. Defaults to 0.5.

        Returns:
            List: A list of weighted squared errors between the predicted values from the SVD and NMF models.

        This function calculates the weighted squared errors between the predicted values from the SVD and NMF models. It iterates over the corresponding predictions from both models and calculates the squared error for each pair of predictions. The weighted squared errors are then returned as a list.

        Note:
            - The weighted squared errors are calculated as (pred_svd.est * weight_svd - pred_nmf.est * weight_nmf)**2.
            - The weights are used to assign different importance to the predictions from each model.

        """
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
        """
        Predicts recommendations for a given user ID using the specified algorithm.

        Args:
            userId (int): The ID of the user for whom recommendations are to be predicted.
            items (List): A list of items for which recommendations are to be predicted.
            algo: The algorithm used for prediction.

        Returns:
            List: A list of predictions sorted in descending order based on the estimated value.

        Description:
            This function takes a user ID, a list of items, and an algorithm as input. It iterates over each item in the list and predicts the rating for the user on that item using the specified algorithm. The predicted ratings are stored in a list along with the corresponding item IDs. The list of predictions is then sorted in descending order based on the estimated value. The sorted list of predictions is returned.

        Note:
            - The algorithm should have a predict method that takes a user ID and an item ID as input and returns a prediction object with properties iid (item ID) and est (estimated value).
            - The item IDs are stored in a separate list to avoid duplicate predictions.
        """
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
        """
        Generates parallel recommendations for a given user ID using multiple random starting weights.

        Args:
            userId (int): The ID of the user for whom recommendations are generated.
            totalParalel (int): The number of parallel processes to use for recommendation generation.

        Returns:
            list: A sorted list of tuples, where each tuple contains the result of a parallel recommendation generation process.
                  The tuple consists of a string representing the result key and a dictionary containing the following keys:
                      - "w1" (float): The value of w1 for the corresponding random starting weight.
                      - "w2" (float): The value of w2 for the corresponding random starting weight.
                      - "rmse" (float): The root mean squared error (RMSE) value of the recommendation generation process.
                      - "preds_svd" (list): The predicted ratings for the user on items using the SVD model.
                      - "preds_nmf" (list): The predicted ratings for the user on items using the NMF model.

        Note:
            - The function uses the `adam_optimizer` method to optimize the random starting weights.
            - The function uses the `predictRecommendation` method to generate predictions using the SVD and NMF models.
            - The function uses the `calculateError` method to calculate the error between the predicted ratings from the SVD and NMF models.
            - The function uses the `calculateRMSE` method to calculate the RMSE value of the recommendation generation process.
        """
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
        """
        Retrieves recommendations for a given user ID.

        Args:
            userId (int): The ID of the user for whom recommendations are to be retrieved.
            topN (int): The maximum number of recommendations to retrieve.
            totalParallel (int, optional): The number of parallel processes to use for recommendation generation. Defaults to 2.

        Returns:
            pandas.DataFrame: A DataFrame containing the concatenated recommendations from both the SVD and NMF models.

        Raises:
            None
        """
        results = self.parallelRecommendation(userId, totalParallel)[0]
        w1 = topN *(1- results[1]["w1"] )
        w2 = topN *(1- results[1]["w2"] ) 
        print(w1, w2)
        takeSvd: int = int(np.ceil(w1))
        takeNMF: int = int(np.ceil(w2))
        print(f"takeSvd: {takeSvd}, takeNMF: {takeNMF}")
        predsSVDTop = results[1]["preds_svd"][:takeSvd]
        predsNMFTop = results[1]["preds_nmf"][:takeNMF]
        dataFrameSVD = pd.DataFrame(predsSVDTop)
        dataFrameNMF = pd.DataFrame(predsNMFTop)
        dfReturn = pd.concat([dataFrameSVD, dataFrameNMF], ignore_index=True)
        dfReturn.drop(columns=["details","r_ui"], inplace=True)
        return dfReturn
    def modelRMSE(self):
        print(rmse(self.svdModel.test(self.test_set)))
        print(rmse(self.nmfModel.test(self.test_set)))
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