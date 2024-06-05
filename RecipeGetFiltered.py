import pandas as pd
import numpy as np
def getFilteredRecipes(datasetPathReviews: str, datasetPathRecipes: str):
    dfRecipes = pd.read_csv(datasetPathRecipes)
    dfReviews = pd.read_csv(datasetPathReviews)
    
    dfRecipeFiltered = pd.DataFrame(columns=dfRecipes.columns)
    for indx, column in dfRecipes.iterrows():
        row = column["RecipeId"] 
        print(row)
        break
    return dfRecipeFiltered