import pandas as pd
import numpy as np
def getFilteredRecipes(datasetPathReviews: str, datasetPathRecipes: str):
    """
    Generates a filtered DataFrame of recipes based on the given dataset paths for reviews and recipes.

    Parameters:
        datasetPathReviews (str): The path to the CSV file containing the reviews dataset.
        datasetPathRecipes (str): The path to the CSV file containing the recipes dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered recipes.

    """
    dfRecipes = pd.read_csv(datasetPathRecipes)
    dfReviews = pd.read_csv(datasetPathReviews)
    
    dfRecipeFiltered = pd.DataFrame(columns=dfRecipes.columns)
    for indx, column in dfRecipes.iterrows():
        row = column["RecipeId"] 
        print(row)
        break
    return dfRecipeFiltered