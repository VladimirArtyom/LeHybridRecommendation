import pandas as pd
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from typing import List, Dict
from enum import Enum


class ContentBasedEnum(Enum):
    CV_Content = "Count_Vectorizer_Content"
    CV_Metadata = "Count_Vectorizer_Metadata"
    E_Content = "Embedding_Content"
    PRE_Content = "PreTrained_Content"


class ContentBasedRecommender():
    
    def __init__(self, dataset: str, columns: List[str]):
        self.df: pd.DataFrame = pd.read_csv(dataset)
        self.recipeIndices: pd.DataFrame = None
        self.column_of_interest: List[str] = columns
        
        self.pretrained_model_content = None
        self.embedding_content = None
        self.countVectorizer_metadata = None
        self.countVectorizer_content = None
        
        self.pretrained_model_embedding = None
        self.embedding_model = None
        self.cv_model_metadata = None
        self.cv_model_content = None

        self.pretrained_model_similarity = None
        self.embedding_similarity = None
        self.cv_content_similarity = None
        self.cv_metadata_similarity = None

        self.savePathEmbeddingCVMetadata: str = "caches/Embedding_CV_Metadata.pkl"
        self.savePathEmbeddingCVContent: str = "caches/Embedding_CV_Content.pkl"
        self.savePathEmbeddingGlove: str = "caches/Embedding_Glove.pkl"
        self.savePathEmbeddingPretrained: str = "caches/Embedding_PreTrained.pkl"

        self.savePathSimilarityCVMetadata:str = "caches/Similarity_CV_Metadata.pkl"
        self.savePathSimilarityCVContent:str = "caches/Similarity_CV_Content.pkl"
        self.savePathSimilarityGlove: str = "caches/Similarity_Glove.pkl"
        self.savePathSimilarityPretrained: str = "caches/Similarity_PreTrained.pkl"

    def prepareRecipeIndices(self, column_name: str):
        df_copy = self.df.copy().reset_index()
        df_copy = df_copy.set_index(column_name)
        self.recipeIndices = pd.Series(df_copy["index"], index=df_copy.index)

    def preprocess_dataset(self):
        self.df.fillna("[]", inplace=True)
        self.df = self.df[self.column_of_interest].copy()
        #self.df[self.column_of_interest[0]] = self.df[self.column_of_interest[0]].apply(self.lower_case)
        self.df[self.column_of_interest[1]] = self.df[self.column_of_interest[1]].apply(self.lower_case)
        self.df[self.column_of_interest[2]] = self.df[self.column_of_interest[2]].apply(self.lower_case)
        self.df[self.column_of_interest[3]] = self.df[self.column_of_interest[3]].apply(self.lower_case)
        self.df[self.column_of_interest[4]] = self.df[self.column_of_interest[4]].apply(self.mergeAuthorName)
        self.df[self.column_of_interest[5]] = self.df[self.column_of_interest[5]].apply(self.cStrToList).apply(self.listTostr)
        self.df[self.column_of_interest[6]] = self.df[self.column_of_interest[6]].apply(self.listTostr)
        self.df["contents"] = self.df["Description"] + self.df["RecipeInstructions"] 
        self.df["contents_metadata"] = self.df["AuthorName"] + self.df["Keywords"] +  self.df["RecipeCategory"] + self.df["TotalTime"]

    def sentence_embedding(self, sentence: str):
        tokens = sentence.lower().split()
        vectors = [self.embedding_model[word] for word in tokens if word in self.embedding_model]

        return np.mean(vectors, axis=0)

    def setupPretrainedModel(self, modelName: str):
        self.pretrained_model_content = SentenceTransformer(modelName)

    def setupEmbedding(self, glove_name: str):
        self.embedding_model = KeyedVectors.load_word2vec_format(glove_name, binary=False, no_header=True)

    def setupCountVectorizer(self):
        self.cv_model_metadata = CountVectorizer(stop_words='english', lowercase=True)
        self.cv_model_content = CountVectorizer(stop_words='english', lowercase=True)
    
    def setup(self, glove_name: str, modelName: str):
        self.setupEmbedding(glove_name)
        self.setupCountVectorizer()
        self.setupPretrainedModel(modelName)
        print("Setup Completed")

    def fitPretrainedModel(self, column):
        self.pretrained_model_embedding = self.pretrained_model_content.encode(self.df.loc[:, column].values)

    def fitEmbedding(self, column: str):
        embedding_arrays: np.ndarray = self.df.loc[:, column].apply(self.sentence_embedding).values
        embedding_arrays = np.array([embed for embed in embedding_arrays])
        self.embedding_content = embedding_arrays

    def fitCountVectorizer(self, column: str):
        if (column == "contents"):
           self.countVectorizer_content = self.cv_model_content.fit_transform(self.df["contents"])
        elif(column == "contents_metadata"):
           self.countVectorizer_metadata = self.cv_model_metadata.fit_transform(self.df["contents_metadata"])
    
    def fit(self, column_1: str, column_2: str,
            isUsingEmbedding: bool = False, isUsingPretrainedModel: bool = False):
        if self.checkCache(self.savePathEmbeddingCVContent) == False:
            self.fitCountVectorizer(column_1) # CV for Contents
            self.saveCache(self.savePathEmbeddingCVContent, self.countVectorizer_content)
        else:
            self.countVectorizer_content = self.loadCache(self.savePathEmbeddingCVContent)
        
        if self.checkCache(self.savePathEmbeddingCVMetadata) == False:
            self.fitCountVectorizer(column_2) # CV for Metadata
            self.saveCache(self.savePathEmbeddingCVMetadata, self.countVectorizer_metadata)
        else:
            self.countVectorizer_metadata = self.loadCache(self.savePathEmbeddingCVMetadata)
        if (self.checkCache(self.savePathEmbeddingGlove) == False and isUsingEmbedding):
            self.fitEmbedding(column_1)
            self.saveCache(self.savePathEmbeddingGlove, self.embedding_content)
        else:
            self.embedding_content = self.loadCache(self.savePathEmbeddingGlove)

        if (self.checkCache(self.savePathEmbeddingPretrained) ==False and isUsingPretrainedModel):
            self.fitPretrainedModel(column_1)
            self.saveCache(self.savePathEmbeddingPretrained, self.pretrained_model_embedding)
        else:
            self.pretrained_model_embedding = self.loadCache(self.savePathEmbeddingPretrained)

    def checkCache(self, path: str):
        if os.path.exists(path):
            return True
        return False

    def saveCache(self, path: str, objectFile):
        with open(path, 'wb') as f:
            pickle.dump(objectFile, f)

    def loadCache(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def calculate_similarity(self, isUsingEmbedding: bool = False, isUsingPretrainedModel: bool = False):
        if (self.checkCache(self.savePathSimilarityCVContent) == False):
            self.cv_content_similarity = cosine_similarity(self.countVectorizer_content,
                                                        self.countVectorizer_content)
            self.saveCache(self.savePathSimilarityCVContent, self.cv_content_similarity)
        else :
            self.cv_content_similarity = self.loadCache(self.savePathSimilarityCVContent)

        
        if (self.checkCache(self.savePathSimilarityCVMetadata) == False):
            self.cv_metadata_similarity = cosine_similarity(self.countVectorizer_metadata,
                                                            self.countVectorizer_metadata)
            self.saveCache(self.savePathSimilarityCVMetadata, self.cv_metadata_similarity)
        else:
            self.cv_metadata_similarity = self.loadCache(self.savePathSimilarityCVContent)

        if (self.checkCache(self.savePathSimilarityGlove) == False and isUsingEmbedding):
            self.embedding_similarity = cosine_similarity(self.embedding_content, self.embedding_content)
            self.saveCache(self.savePathSimilarityGlove, self.embedding_similarity)

        else:
            self.embedding_similarity = self.loadCache(self.savePathSimilarityGlove)
        
        if self.checkCache(self.savePathSimilarityPretrained) == False and isUsingPretrainedModel:
            self.pretrained_model_similarity = cosine_similarity(self.pretrained_model_embedding,
                                                                self.pretrained_model_embedding)
            self.saveCache(self.savePathSimilarityPretrained, self.pretrained_model_similarity)
        else: 
            self.pretrained_model_similarity = self.loadCache(self.savePathSimilarityPretrained)
        print("Similarity matrix is generated!")

    def convertToRecipesId(self, listRecipes: List[int]):
        realRecipes : List = []
        for value in (listRecipes):
            realRecipes.append(self.recipeIndices[self.recipeIndices.values == value].index[0])
        return realRecipes


    def get_recommendation(self, inputRecipe, topN: int , algorithm: ContentBasedEnum = ContentBasedEnum.CV_Content):
        #inputRecipe = inputRecipe.lower()
        indexRecipe = self.recipeIndices[inputRecipe]
        similarity = []
        if (algorithm == ContentBasedEnum.CV_Metadata):
            similarity = list(enumerate(self.cv_metadata_similarity[indexRecipe]))
        elif (algorithm == ContentBasedEnum.CV_Content):
            similarity = list(enumerate(self.cv_content_similarity[indexRecipe]))
        elif (algorithm == ContentBasedEnum.E_Content and self.embedding_similarity is not None):
            similarity = list(enumerate(self.embedding_similarity[indexRecipe]))
        elif (algorithm == ContentBasedEnum.PRE_Content and self.pretrained_model_similarity is not None):
            similarity = list(enumerate(self.pretrained_model_similarity[indexRecipe]))
        similarity = sorted(similarity, key=lambda x: np.average(x[1]), reverse=True)[:topN]

        recommendationRecipes: List[str] = [i[0] for i in similarity]
        realRecipesId = self.convertToRecipesId(recommendationRecipes)
        return realRecipesId
    
    def cleanStr(self, x : str):
        x = x.strip()
        x = x.replace("'","")
        x = x.replace('"',"")
        x.lower()
        return x

    def cStrToList(self, x: str):
        x = x[2:-1]
        x = x.replace('"', "" )
        return [self.cleanStr(s) for s in x.split(',')]

    def listTostr(self,t):
        mainStr: str = ""
        for indx in range(len(t)):
            mainStr += " " + t[indx].lower()
        return mainStr
    def lower_case(self, t: str):
        return t.lower()

    def mergeAuthorName(self, x: str):
        x = x.lower()
        return x.strip().replace(" ", "-").replace(".","")

""" 
if __name__ == "__main__":
    column_of_interest = ["Name", "Description", "RecipeCategory",
                          "TotalTime", "AuthorName", "RecipeInstructions",
                          "Keywords"]
    datasetPath: str = "recipes_filtered.csv"
    glove_file: str = "GoogleNews-vectors-negative300.bin"    

    contentBased = ContentBasedRecommender(datasetPath, column_of_interest)
    contentBased.preprocess_dataset()
    contentBased.prepareRecipeIndices("Name")
    contentBased.setup(glove_file)
    contentBased.fit("contents", "contents_metadata", "all-MiniLM-L12-v2")
    contentBased.calculate_similarity()
"""
