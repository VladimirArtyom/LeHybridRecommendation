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
        """
        Initializes a new instance of the ContentBasedRecommender class.

        Parameters:
            dataset (str): The path to the dataset file.
            columns (List[str]): The list of column names to be used for recommendation.

        Returns:
            None
        """
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
        """
        Prepare recipe indices based on the given column name.

        Args:
            column_name (str): The name of the column to use as the index.

        Returns:
            None

        This function creates a copy of the DataFrame, resets the index, and sets the specified column as the new index. It then creates a new Series object with the index of the DataFrame and assigns it to the `recipeIndices` attribute.

        Example usage:
        ```
        recipe_indices = RecipeIndices()
        recipe_indices.prepareRecipeIndices("recipe_name")
        ```
        """
        df_copy = self.df.copy().reset_index()
        df_copy = df_copy.set_index(column_name)
        self.recipeIndices = pd.Series(df_copy["index"], index=df_copy.index)

    def preprocess_dataset(self):
        """
        Preprocesses the dataset by filling missing values with "[]", selecting the columns of interest,
        applying lowercase to specific columns, merging author name, converting string lists to strings,
        and creating new columns for content and metadata.

        Parameters:
            None

        Returns:
            None
        """
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
        """
        Generate the embedding vector for a given sentence.

        Parameters:
            sentence (str): The input sentence to be embedded.

        Returns:
            numpy.ndarray: The mean embedding vector of the input sentence.

        This function takes a sentence as input and performs the following steps:
        1. Tokenizes the sentence into individual words.
        2. Converts all the words to lowercase.
        3. Checks if each word is present in the pre-trained embedding model.
        4. Retrieves the embedding vector for each word that exists in the model.
        5. Calculates the mean of the retrieved embedding vectors.
        6. Returns the mean embedding vector as a numpy array.

        Example usage:
        ```
        sentence = "I love dogs"
        embedding = sentence_embedding(sentence)
        ```
        """
        tokens = sentence.lower().split()
        vectors = [self.embedding_model[word] for word in tokens if word in self.embedding_model]

        return np.mean(vectors, axis=0)

    def setupPretrainedModel(self, modelName: str):
        """
        Initializes the pretrained model for content embedding.

        Parameters:
            modelName (str): The name of the pretrained model to be used for content embedding.

        Returns:
            None
        """

    def setupEmbedding(self, glove_name: str):
        """
        Initializes the embedding model using the given GloVe file.

        Parameters:
            glove_name (str): The path to the GloVe file.

        Returns:
            None
        """
        self.pretrained_model_content = SentenceTransformer(modelName)

    def setupEmbedding(self, glove_name: str):
        """
        Initializes the embedding model using the given GloVe file.

        Args:
            glove_name (str): The path to the GloVe file.

        Returns:
            None
        """
        self.embedding_model = KeyedVectors.load_word2vec_format(glove_name, binary=False, no_header=True)

    def setupCountVectorizer(self):
        """
        Initializes the count vectorizer models for metadata and content.

        This function creates two count vectorizer models, one for metadata and one for content.
        The models are initialized with the following parameters:
        - stop_words: 'english' - removes common English stop words from the input text.
        - lowercase: True - converts all characters to lowercase before tokenizing.

        Returns:
            None
        """
        self.cv_model_metadata = CountVectorizer(stop_words='english', lowercase=True)
        self.cv_model_content = CountVectorizer(stop_words='english', lowercase=True)
    
    def setup(self, glove_name: str, modelName: str):
        """
        Initializes the recommender system by setting up the embedding model, count vectorizer, and pretrained model.

        Parameters:
            glove_name (str): The name of the GloVe file to use for the embedding model.
            modelName (str): The name of the pretrained model to use.

        Returns:
            None
        """
        self.setupEmbedding(glove_name)
        self.setupCountVectorizer()
        self.setupPretrainedModel(modelName)
        print("Setup Completed")

    def fitPretrainedModel(self, column):
        """
        Fits the pretrained model to the given column of the DataFrame.

        Parameters:
            column (str): The name of the column in the DataFrame to fit the pretrained model to.

        Returns:
            None

        This function encodes the values in the specified column of the DataFrame using the pretrained model and stores the resulting embedding in the `self.pretrained_model_embedding` attribute.
        """
        self.pretrained_model_embedding = self.pretrained_model_content.encode(self.df.loc[:, column].values)


    def fitEmbedding(self, column: str):
        """
        Fits an embedding model to the given column of the DataFrame.

        Parameters:
            column (str): The name of the column in the DataFrame to fit the embedding model to.

        Returns:
            None

        This function applies the `sentence_embedding` function to each value in the specified column of the DataFrame and stores the resulting embeddings in the `self.embedding_content` attribute.
        """
        embedding_arrays: np.ndarray = self.df.loc[:, column].apply(self.sentence_embedding).values
        embedding_arrays = np.array([embed for embed in embedding_arrays])
        self.embedding_content = embedding_arrays

    def fitCountVectorizer(self, column: str):
        """
        Fits the count vectorizer to the specified column of the DataFrame.

        Parameters:
            column (str): The name of the column in the DataFrame to fit the count vectorizer to.
                Must be either "contents" or "contents_metadata".

        Returns:
            None

        This function fits the count vectorizer to the specified column of the DataFrame and stores the resulting
        sparse matrix in the appropriate attribute (`self.countVectorizer_content` for "contents" and
        `self.countVectorizer_metadata` for "contents_metadata").

        Raises:
            ValueError: If the `column` parameter is not either "contents" or "contents_metadata".
        """
        if (column == "contents"):
           self.countVectorizer_content = self.cv_model_content.fit_transform(self.df["contents"])
        elif(column == "contents_metadata"):
           self.countVectorizer_metadata = self.cv_model_metadata.fit_transform(self.df["contents_metadata"])
    
    def fit(self, column_1: str, column_2: str,
            isUsingEmbedding: bool = False, isUsingPretrainedModel: bool = False):
        """
        Fits the count vectorizer, embedding, and pretrained model to the specified columns of the DataFrame.

        Parameters:
            column_1 (str): The name of the first column in the DataFrame to fit the count vectorizer, embedding, and pretrained model to.
            column_2 (str): The name of the second column in the DataFrame to fit the count vectorizer to.
            isUsingEmbedding (bool, optional): Whether to use the embedding model. Defaults to False.
            isUsingPretrainedModel (bool, optional): Whether to use the pretrained model. Defaults to False.

        Returns:
            None

        This function first checks if the count vectorizer for the contents column is already cached. If not, it fits the count vectorizer to the specified column and saves it to the cache. If it is cached, it loads the count vectorizer from the cache.

        Then, it checks if the count vectorizer for the metadata column is already cached. If not, it fits the count vectorizer to the specified column and saves it to the cache. If it is cached, it loads the count vectorizer from the cache.

        If the embedding is not cached and `isUsingEmbedding` is True, it fits the embedding model to the specified column and saves it to the cache. If it is cached, it loads the embedding from the cache.

        If the pretrained model is not cached and `isUsingPretrainedModel` is True, it fits the pretrained model to the specified column and saves it to the cache. If it is cached, it loads the pretrained model from the cache.
        """
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
        """
        Check if a file or directory exists at the given path.

        Parameters:
            path (str): The path to the file or directory.

        Returns:
            bool: True if the file or directory exists, False otherwise.
        """
        if os.path.exists(path):
            return True
        return False

    def saveCache(self, path: str, objectFile):
        """
        Save an object to a file using pickle serialization.

        Parameters:
            path (str): The path to the file where the object will be saved.
            objectFile (object): The object to be saved.

        Returns:
            None
        """
        with open(path, 'wb') as f:
            pickle.dump(objectFile, f)

    def loadCache(self, path: str):
        """
        Load a pickled object from the specified file path.

        Parameters:
            path (str): The path to the file containing the pickled object.

        Returns:
            The unpickled object.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def calculate_similarity(self, isUsingEmbedding: bool = False, isUsingPretrainedModel: bool = False):
        """
        Calculates the similarity between different types of embeddings and saves the similarity matrices to cache.
        
        Parameters:
            isUsingEmbedding (bool, optional): Whether to use the embedding model. Defaults to False.
            isUsingPretrainedModel (bool, optional): Whether to use the pretrained model. Defaults to False.
        
        Returns:
            None
        
        This function calculates the cosine similarity between different types of embeddings and saves the similarity matrices to cache.
        If the similarity matrix for count vectorizer content is not cached, it calculates the cosine similarity between the count vectorizer content and saves it to cache.
        If the similarity matrix for count vectorizer metadata is not cached, it calculates the cosine similarity between the count vectorizer metadata and saves it to cache.
        If the similarity matrix for embedding content is not cached and isUsingEmbedding is True, it calculates the cosine similarity between the embedding content and saves it to cache.
        If the similarity matrix for pretrained model embedding is not cached and isUsingPretrainedModel is True, it calculates the cosine similarity between the pretrained model embedding and saves it to cache.
        Finally, it prints "Similarity matrix is generated!" to indicate that the similarity matrices have been generated.
        """
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
        """
        Converts a list of recipe IDs to their corresponding real recipe IDs.

        Parameters:
            listRecipes (List[int]): A list of recipe IDs.

        Returns:
            List: A list of real recipe IDs corresponding to the input recipe IDs.
        """
        realRecipes : List = []
        for value in (listRecipes):
            realRecipes.append(self.recipeIndices[self.recipeIndices.values == value].index[0])
        return realRecipes


    def get_recommendation(self, inputRecipe, topN: int , algorithm: ContentBasedEnum = ContentBasedEnum.CV_Content):
        """
        Retrieves content-based recommendations for a given input recipe.

        Args:
            inputRecipe (str): The input recipe for which recommendations are to be generated.
            topN (int): The maximum number of recommendations to retrieve.
            algorithm (ContentBasedEnum, optional): The algorithm to use for generating recommendations. 
                                                   Defaults to ContentBasedEnum.CV_Content.

        Returns:
            List[int]: A list of recipe IDs representing the content-based recommendations.

        Raises:
            None

        Algorithm:
            The function first retrieves the index of the input recipe from the recipeIndices dictionary. 
            Then, it calculates the similarity between the input recipe and other recipes based on the specified algorithm. 
            The similarity is calculated using the cosine similarity between the count vectorizer representations of the recipes. 
            The function then sorts the similarity values in descending order and selects the topN recipes with the highest similarity. 
            The function finally converts the recipe names to recipe IDs and returns the list of recommended recipe IDs.
        """
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
        """
        Cleans the input string by removing leading and trailing whitespace, 
        single quotes, and double quotes. Converts the string to lowercase 
        and returns the cleaned string.

        Parameters:
            x (str): The input string to be cleaned.

        Returns:
            str: The cleaned string.
        """
        x = x.strip()
        x = x.replace("'","")
        x = x.replace('"',"")
        x.lower()
        return x

    def cStrToList(self, x: str):
        """
        Converts a string representation of a list into an actual list.

        Parameters:
            x (str): The string representation of the list.

        Returns:
            list: The converted list.

        Example:
            >>> cStrToList('["apple", "banana", "cherry"]')
            ['apple', 'banana', 'cherry']
        """
        x = x[2:-1]
        x = x.replace('"', "" )
        return [self.cleanStr(s) for s in x.split(',')]

    def listTostr(self,t):
        """
        Convert a list of strings into a single string by concatenating all the strings in the list.

        Parameters:
            t (List[str]): The list of strings to be converted.

        Returns:
            str: The concatenated string.
        """
        mainStr: str = ""
        for indx in range(len(t)):
            mainStr += " " + t[indx].lower()
        return mainStr
    
    def lower_case(self, t: str):
        """
        Converts the given string to lowercase.

        Args:
            t (str): The string to be converted to lowercase.

        Returns:
            str: The lowercase version of the input string.
        """
        return t.lower()

    def mergeAuthorName(self, x: str):
        """
        Merges the given author name by converting it to lowercase, removing leading and trailing whitespace,
        replacing spaces with hyphens, and removing dots.

        Parameters:
            x (str): The author name to be merged.

        Returns:
            str: The merged author name.
        """
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
