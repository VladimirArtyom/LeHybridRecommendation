import importlib
import Integrate_ContentRC
import Integrate_SurpriseCollaborativeRC
import Integrate_DeepLearningCollaborativeRC
import numpy as np
import pandas as pd
importlib.reload(Integrate_ContentRC)
importlib.reload(Integrate_SurpriseCollaborativeRC)
importlib.reload(Integrate_DeepLearningCollaborativeRC)


from Integrate_DeepLearningCollaborativeRC import EmbeddingNetwork, DeepLearningCollaborativeBasedRecommender
from Integrate_SurpriseCollaborativeRC import SurpriseCollaborativeBasedRecommender
from Integrate_ContentRC import ContentBasedEnum
from Integrate_ContentRC import ContentBasedRecommender
from typing import List, Tuple

np.random.seed(0)
class HybridRecommender:

    def __init__(self, seed: int, path="./recipes_filtered.csv"):
        self.contentBasedRecommender = None
        self.supriseCFRecommender = None 
        self.deeplearningCFRecommender = None
        self.reviewDataset = pd.read_csv(path)
        self.reviewDataset.set_index("RecipeId", inplace=True)
    
    def setupContentBasedRecommender(self, datasetPath: str,
                                    useCols: List[str], gloveFile="Embeddings/glove.6B.50d.txt",
                                    modelName="all-MiniLM-L12-v2", isEmbedding = True, isPretrained= True):
        self.contentBasedRecommender = ContentBasedRecommender(datasetPath, useCols)
        self.contentBasedRecommender.preprocess_dataset()
        self.contentBasedRecommender.prepareRecipeIndices("RecipeId")
        self.contentBasedRecommender.setup(gloveFile, modelName)
        self.contentBasedRecommender.fit("contents", "contents_metadata", isUsingEmbedding=isEmbedding, isUsingPretrainedModel=isPretrained)
        self.contentBasedRecommender.calculate_similarity(isUsingEmbedding=isEmbedding,
                                                          isUsingPretrainedModel=isPretrained)

    def setupSurpriseCollaborativeRecommender(self, datasetPath: str, useCols: List[str],
                                               ratingScale: Tuple, targetAlign = ["AuthorId", "RecipeId", "Rating"]):
        self.supriseCFRecommender = SurpriseCollaborativeBasedRecommender(datasetPath,
                                                                         ratingScale,
                                                                         useCols)
        self.supriseCFRecommender.alignDataset(targetAlign)
        self.supriseCFRecommender.splitDataset()
        self.supriseCFRecommender.fit(0.5, 0.5)

    def setupDeepLearningCollaborativeRecommender(self, datasetPath: str, useCols: List[str], 
                                                  networkSize: int = 150,
                                                  targetAlign = ["AuthorId", "RecipeId", "Rating"],
                                                  isSaveModel: bool = True, modelPath = "caches/model.pt"):
        self.deeplearningCFRecommender = DeepLearningCollaborativeBasedRecommender(datasetPath, useCols)
        self.deeplearningCFRecommender.alignDataset(targetAlign)
        self.deeplearningCFRecommender.createDataset()
        self.deeplearningCFRecommender.splitDataset(0.1)
        self.deeplearningCFRecommender.setupNetwork(networkSize)
        modelDict, history, lrHistory = self.deeplearningCFRecommender.trainNetwork()
        if isSaveModel == True:
            self.deeplearningCFRecommender.saveModel(modelDict, history, lrHistory, modelPath)
        self.deeplearningCFRecommender.modelRMSE("cpu")

    def getRecommendations(self, userId: int, topN: int = 10, parralel: int = 2):
        """
        Retrieves recommendations for a given user ID.

        Args:
            userId (int): The ID of the user for whom recommendations are to be retrieved.
            topN (int, optional): The maximum number of recommendations to retrieve. Defaults to 10.
            parralel (int, optional): The number of parallel processes to use for recommendation generation. Defaults to 2.

        Returns:
            pandas.DataFrame: A DataFrame containing the concatenated recommendations from both the deep learning collaborative filtering recommender and the Surprise collaborative filtering recommender.

        Raises:
            ValueError: If the user ID is greater than or equal to 1821.
        """
        if(userId < 1821):
            recommendationDL = self.deeplearningCFRecommender.getRecommendation(userId, topN)
            recommendationSurprise = self.supriseCFRecommender.getRecommendations(userId, topN, parralel)
            allRecommendations = pd.concat([recommendationDL[["uid","iid","est"]], recommendationSurprise], ignore_index=True)
        
            return allRecommendations
        else:
            raise ValueError("User Id should be less than 1821")

    def getContentBasedRecommendations(self, recipeIds: List[int], topN: int, ):
        """
        Retrieves content-based recommendations for a list of recipe IDs.

        Args:
            recipeIds (List[int]): A list of recipe IDs for which recommendations are to be retrieved.
            topN (int): The maximum number of recommendations to retrieve for each recipe ID.

        Returns:
            List[int]: A list of recipe IDs representing the content-based recommendations.

        Raises:
            None
        """
        allRecommendations = set()
        for  indx, recipeId in enumerate(recipeIds):
            embedsRecommender = self.contentBasedRecommender.get_recommendation(recipeId, int(topN* 0.4), ContentBasedEnum.E_Content)
            preRecommender = self.contentBasedRecommender.get_recommendation(recipeId, int(topN * 0.3), ContentBasedEnum.E_Content)
            countRecommender = self.contentBasedRecommender.get_recommendation(recipeId, int(topN * 0.2), ContentBasedEnum.CV_Content)
            countRecommenderMetadata = self.contentBasedRecommender.get_recommendation(recipeId, int(topN * 0.1), ContentBasedEnum.CV_Metadata)
            
            allRecommendations = allRecommendations.union(embedsRecommender) 
            allRecommendations = allRecommendations.union(preRecommender)
            allRecommendations = allRecommendations.union(countRecommender)
            allRecommendations = allRecommendations.union(countRecommenderMetadata)
            recipeIds = np.insert(recipeIds, indx+1, list(allRecommendations))

        return recipeIds

    def showRecommendation(self, indexRecommendations: List[int], indexRecommendationSurprise: List[int]):
        recommendationWithContent = self.reviewDataset.loc[indexRecommendations].drop_duplicates()
        recommendationSurprise = self.reviewDataset[self.reviewDataset.index.isin(indexRecommendationSurprise)].drop_duplicates()

        return pd.concat([recommendationWithContent, recommendationSurprise], ignore_index=True)

    def showRecommendations(self, userId: int, topN: int = 10, parralel: int = 2):
        CFRecommendations = self.getRecommendations(userId, topN, parralel)


            
