import pandas as pd
import numpy as np

class HybridRecommender:
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