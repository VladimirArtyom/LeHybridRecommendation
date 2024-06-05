import pandas as pd
import numpy as np
import torch
import math
from itertools import zip_longest

import pickle
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Dict, Union
import copy


class RecipesIterator:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """
        Initializes a RecipesIterator object.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target data.
            batch_size (int, optional): The batch size for iterating over the data. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            None
        """
        X, y = np.asarray(X), np.asarray(y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]
        
        self.X = X 
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(np.ceil(X.shape[0] // self.batch_size))
        self._current = 0

    def __iter__(self):
        """
        Returns the iterator object itself.

        This method is part of the iterator protocol in Python. It allows an object to be iterated over using a for loop. When the __iter__() method is called on an object, it should return an iterator object. In this case, the __iter__() method of the RecipesIterator class returns the object itself, allowing it to be used in a for loop.

        Returns:
            RecipesIterator: The iterator object itself.
        """
        return self
    
    def __next__(self):
        """
        Returns the next item from the iterator.

        This method is part of the iterator protocol in Python. It is called when the next item in the iterator is requested. It should return the next item in the iteration or raise the StopIteration exception to indicate the end of the iteration.

        Returns:
            The next item in the iteration.

        Raises:
            StopIteration: If there are no more items to iterate over.

        """
        return self.next()
    
    def next(self):
        """
        Returns the next batch of data from the iterator.

        This method is used to iterate over the data in batches. It returns a tuple containing the next batch of input data (`X`) and the corresponding target data (`y`). The batch size is determined by the `batch_size` attribute of the object.

        Returns:
            tuple: A tuple containing the next batch of input data (`X`) and the corresponding target data (`y`).

        Raises:
            StopIteration: If there are no more batches to iterate over.
        """
        if self._current >= self.n_batches:
            raise StopIteration()
        
        start = self._current
        bs = self.batch_size
        self._current += 1
        # Starting from [0: 1*32] , next [1*32: 2*32], etc
        return self.X[start*bs:(start + 1)*bs], self.y[start*bs:(start + 1)*bs]

class EmbeddingNetwork(nn.Module):
    
    def __init__(self, nUsers: int, nRecipes: int, nFactors: int =50,
                 embeddingDropout: float=0.02, hiddenLayers: Union[List, float] = 10,
                 dropout: Union[List, float] = 0.2):
        """
        Initializes a new instance of the EmbeddingNetwork class.

        Args:
            nUsers (int): The number of users.
            nRecipes (int): The number of recipes.
            nFactors (int, optional): The number of factors. Defaults to 50.
            embeddingDropout (float, optional): The dropout rate for the embedding layer. Defaults to 0.02.
            hiddenLayers (Union[List, float], optional): The number of hidden layers or the size of each hidden layer. Defaults to 10.
            dropout (Union[List, float], optional): The dropout rate for each hidden layer. Defaults to 0.2.

        Initializes the following attributes:
            - hidden (List): A list of hidden layer sizes.
            - dropouts (List): A list of dropout rates for each hidden layer.
            - last_hidden (int): The size of the last hidden layer.
            - users (nn.Embedding): An embedding layer for users.
            - recipes (nn.Embedding): An embedding layer for recipes.
            - embedDropout (nn.Dropout): A dropout layer for the embedding layer.
            - hiddenLayers (nn.Sequential): A sequential container for the hidden layers.
            - output (nn.Linear): A linear layer for the output.

        Calls the _init() method.

        Returns:
            None
        """
        super().__init__()
        self.hidden: List = self.makeList(hiddenLayers)
        self.dropouts: List = self.makeList(dropout)
        self.last_hidden = self.hidden[-1]
        
        self.users = nn.Embedding(nUsers, nFactors)
        self.recipes = nn.Embedding(nRecipes, nFactors)
        self.embedDropout = nn.Dropout(embeddingDropout)
        self.hiddenLayers = nn.Sequential(*list(self.layerGenerator(nFactors * 2)))
        self.output = nn.Linear(self.last_hidden, 1)
        self._init()

    def forward(self, users, recipes, minmax=None):
        """
        Computes the forward pass of the model.

        Args:
            users (torch.Tensor): A tensor containing the user indices.
            recipes (torch.Tensor): A tensor containing the recipe indices.
            minmax (tuple, optional): A tuple containing the minimum and maximum ratings. Defaults to None.

        Returns:
            torch.Tensor: A tensor containing the predicted ratings.

        Description:
            This function takes in user and recipe indices as input and computes the forward pass of the model. It first concatenates the embeddings of the users and recipes along the second dimension. Then, it applies dropout to the concatenated features. Next, it passes the features through a sequential container of hidden layers. Finally, it applies the sigmoid activation function to the output of the last hidden layer and scales the predicted ratings based on the minimum and maximum ratings if specified. The function returns the predicted ratings as a tensor.
        """
        features = torch.cat([self.users(users), self.recipes(recipes)], dim=1)
        x = self.embedDropout(features)
        x = self.hiddenLayers(x)
        out = torch.sigmoid(self.output(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out * (max_rating - min_rating + 1) + min_rating - 0.5
        return out
        
    
    def layerGenerator(self, n_in):
        """
        Generates a sequence of layers for a neural network model.

        Args:
            n_in (int): The number of input features.

        Yields:
            torch.nn.Module: A linear layer followed by a ReLU activation function.
            torch.nn.Module: A dropout layer if the corresponding dropout rate is greater than 0.

        """
        for n_out, drop_rate in zip_longest(self.hidden, self.dropouts):
            yield nn.Linear(n_in, n_out)
            yield nn.ReLU()
            if drop_rate is not None and drop_rate > 0:
                yield nn.Dropout(drop_rate)
            n_in = n_out

    def makeList(self, n):
        """
        Convert a single number or an iterable to a list.

        Parameters:
            n (int or float or iterable): The input number or iterable to be converted to a list.

        Returns:
            list: A list containing the input number or elements of the iterable.

        Raises:
            TypeError: If the input is neither a single number nor an iterable.
        """
        if isinstance(n, (int, float)):
            return [n]
        elif hasattr(n, "__iter__"):
            return list(n)
        raise TypeError("Layers configuration must be a single number or a list of numbers")

    def initWeight(self, layer):
        """
        Initializes the weights of a given layer using the Xavier uniform distribution.

        Parameters:
            layer (nn.Linear): The layer to initialize the weights for.

        Returns:
            None
        """
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)

    def _init(self):
        """
        Initializes the weights of the neural network.

        This function initializes the weights of the neural network by sampling them from a uniform distribution between -0.05 and 0.05. The weights of the users and recipes layers are initialized separately. The weights of the hidden layers are initialized using the `initWeight` function. The output layer's weights are also initialized using the `initWeight` function.

        Parameters:
            None

        Returns:
            None
        """
        self.users.weight.data.uniform_(-0.05, 0.05)
        self.recipes.weight.data.uniform_(-0.05, 0.05)
        self.hiddenLayers.apply(self.initWeight)
        self.initWeight(self.output)


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        """
        Initializes the CyclicLR class with the given optimizer, schedule, and last_epoch.

        Parameters:
            optimizer (torch.optim.Optimizer): The optimizer to use for the learning rate scheduling.
            schedule (function): The learning rate schedule function.
            last_epoch (int, optional): The index of the last epoch. Defaults to -1.

        Returns:
            None
        """
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns a list of learning rates calculated using the schedule function for each base learning rate.

        :return: A list of floats representing the learning rates.
        :rtype: List[float]
        """
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
    
class DeepLearningCollaborativeBasedRecommender():
    def __init__(self, datasetPath: str, useCols, seed: int = 0):
        """
        Initializes a new instance of the DeepLearningCollaborativeBasedRecommender class.

        Parameters:
            datasetPath (str): The path to the dataset file.
            useCols: The columns to be used from the dataset.
            seed (int, optional): The seed for random number generation. Defaults to 0.

        Returns:
            None
        """
        self.setupSeed(seed)
        self.df = pd.read_csv(datasetPath, usecols=useCols)
        self.minMax = self.df["Rating"].min().astype(float), self.df["Rating"].max().astype(float)
        self.network: EmbeddingNetwork = None

    def setupSeed(self, seed: int):
        """
        Sets the seed for random number generation in NumPy, PyTorch, and CUDA.

        Parameters:
            seed (int): The seed value to be used for random number generation.

        Returns:
            None
        """
        listOfSeeds : List = [np.random.seed, torch.manual_seed, torch.cuda.manual_seed]
        for l in listOfSeeds:
            l(seed)

    def alignDataset(self, target: List[str]):
        """
        Aligns the dataset by selecting only the specified columns from the dataframe.

        Parameters:
            target (List[str]): A list of column names to select from the dataframe.

        Returns:
            None
        """
        self.df = self.df[target]

    def createDataset(self):
        """
        Creates a dataset by mapping the unique users and recipes in the dataframe to new indices.

        This function retrieves the unique users and recipes from the "AuthorId" and "RecipeId" columns of the dataframe, respectively. It then creates dictionaries that map each unique user and recipe to a new index. The new indices are obtained by enumerating the unique values.

        The function assigns the number of unique users and recipes to the instance variables `nUsers` and `nRecipes`, respectively.

        The function then creates two new columns in the dataframe: "newAuthorId" and "newRecipeId". The values in these columns are obtained by mapping the values in the "AuthorId" and "RecipeId" columns to the corresponding new indices using the dictionaries `authorToIndx` and `recipeToIndx`.

        Finally, the function assigns the values in the "newRecipeId" column to the instance variable `newRecipes`.

        Parameters:
            None

        Returns:
            None
        """
        unique_users: pd.Series = self.df["AuthorId"].unique()
        unique_recipes: pd.Series = self.df["RecipeId"].unique()
        authorToIndx: Dict = {
            authorId: indx for indx, authorId in enumerate(unique_users)
        }

        recipeToIndx: Dict = {
            recipeId: indx for indx, recipeId in enumerate(unique_recipes)
        }

        self.nUsers = len(unique_users)
        self.nRecipes = len(unique_recipes)

        self.df["newAuthorId"] = self.df["AuthorId"].map(authorToIndx)
        self.df["newRecipeId"] = self.df["RecipeId"].map(recipeToIndx)
        self.newRecipes = self.df.loc[:, "newRecipeId"]

    def batches(self,X, y, batchSize: int = 32, shuffle=True):
        """
        Generates batches of input data and corresponding target data for training a model.

        Args:
            X (array-like): The input data.
            y (array-like): The target data.
            batchSize (int, optional): The size of each batch. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Yields:
            tuple: A tuple containing the batch of input data (torch.Tensor) and the batch of target data (torch.Tensor).
        """
        for xb, yb in RecipesIterator(X,
                                      y,
                                      batchSize, shuffle):
            xb : torch.Tensor = torch.LongTensor(xb)
            yb : torch.Tensor = torch.FloatTensor(yb)
            yield xb, yb.view(-1, 1)

    def cosineAnnealing(self, num_epochs_per_cycle, min_lr: float=0):
        """
        Generates a cosine annealing learning rate scheduler.

        Args:
            num_epochs_per_cycle (int): The number of epochs in each cycle.
            min_lr (float, optional): The minimum learning rate. Defaults to 0.

        Returns:
            function: The cosine annealing learning rate scheduler function.

        The cosine annealing learning rate scheduler function takes two arguments:
            - epoch (int): The current epoch.
            - base_lr (float): The base learning rate.

        The function calculates the adjusted learning rate based on the current epoch within the cycle. It uses the formula:
            adjusted_lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * current_epoch_within_cycle / num_epochs_per_cycle)) / 2

        The function returns the adjusted learning rate.
        """
        
        def scheduler(epoch, base_lr: float):
            """
            Calculates the adjusted learning rate based on the current epoch within the cycle.

            Args:
                epoch (int): The current epoch.
                base_lr (float): The base learning rate.

            Returns:
                float: The adjusted learning rate.

            The function calculates the adjusted learning rate using the cosine annealing learning rate scheduler. It takes the current epoch and the base learning rate as input and returns the adjusted learning rate. The adjusted learning rate is calculated using the formula:
                adjusted_lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * current_epoch_within_cycle / num_epochs_per_cycle)) / 2
            """
            currentEpochWithinCycle = epoch % num_epochs_per_cycle 
            adjustedLr = min_lr + (base_lr - min_lr) * ( 1 + math.cos(math.pi * currentEpochWithinCycle / num_epochs_per_cycle)) / 2
            return adjustedLr

        return scheduler

    def splitDataset(self, testSize: float = 0.1):
        """
        Split the dataset into training and validation sets.

        Parameters:
            testSize (float, optional): The proportion of the dataset to be used for validation. Defaults to 0.1.

        Returns:
            None

        This function splits the dataset into training and validation sets using the train_test_split function from the sklearn.model_selection module. The dataset is split based on the provided testSize parameter, with the default value being 0.1. The resulting training and validation sets are stored in the 'datasets' dictionary, with the keys 'train' and 'val' respectively. The sizes of the training and validation sets are stored in the 'dataset_sizes' dictionary.
        """
        X = pd.DataFrame({"AuthorId": self.df["newAuthorId"], "RecipeId": self.df["newRecipeId"]})
        Y = self.df["Rating"].astype(float)
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=testSize, random_state=0)
        self.datasets = {"train": (x_train, y_train), "val": (x_val, y_val)}
        self.dataset_sizes = {"train": len(x_train), "val": len(x_val)}

    def setupNetwork(self, nFactors, hiddenLayers= [500, 500, 500],
                     embeddingDropout = 0.05, dropouts=[0.50, 0.50, 0.35]):
        """
        Set up the network for the collaborative filtering recommendation system.

        Parameters:
            nFactors (int): The number of factors to consider in the collaborative filtering model.
            hiddenLayers (List[int], optional): The number of neurons in each hidden layer of the network. Defaults to [500, 500, 500].
            embeddingDropout (float, optional): The dropout rate for the embedding layer. Defaults to 0.05.
            dropouts (List[float], optional): The dropout rates for each hidden layer. Defaults to [0.50, 0.50, 0.35].

        Returns:
            None
        """
        self.network = EmbeddingNetwork(self.nUsers, self.nRecipes, nFactors,
                                        embeddingDropout, hiddenLayers, dropouts)
    
    def trainNetwork(self, epochs: int = 1000,
                    batchSize: int = 2000, device: str="cpu",
                    lr: float=1e-3, weight_decay: float=0.08, patience: int= 15):
        """
        Trains the network using the specified number of epochs and hyperparameters.

        Parameters:
            epochs (int, optional): The number of epochs to train the network. Defaults to 1000.
            batchSize (int, optional): The batch size for training. Defaults to 2000.
            device (str, optional): The device to use for training. Defaults to "cpu".
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.08.
            patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 15.

        Returns:
            bestWeights (dict): The weights of the network with the best validation loss.
            history (list): A list of dictionaries containing the training and validation loss for each epoch.
            lrHistory (list): A list of floats containing the learning rate for each epoch.
        """
        lrHistory: List[float] = []
        history: List = []
        noImprovement: int = 0
        bestValLoss = np.inf
        bestWeights = None
        self.network.to(device)
        criterion = nn.MSELoss(reduction="sum")
        optimizer = optim.Adam(self.network.parameters(),
                               lr=lr, weight_decay=weight_decay)
        iterationsPerEpoch = int(math.ceil(self.dataset_sizes["train"] / batchSize))
        scheduler = CyclicLR(optimizer, self.cosineAnnealing(num_epochs_per_cycle=iterationsPerEpoch * 2, min_lr=lr/10))

        for epoch in range(epochs):
            stats = {"epoch": epoch + 1, "total": epochs}

            currentTrainLoss: float = 0.0
            currentValLoss: float = 0.0

            for batch in self.batches(*self.datasets["train"], batchSize=batchSize, shuffle=True):
                xBatch, yBatch = [b.to(device) for b in batch]
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.network(xBatch[:, 0], xBatch[:, 1], self.minMax)
                    loss = criterion(outputs, yBatch)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    lrHistory.append(scheduler.get_lr())
                
                currentTrainLoss += loss.item()

            for batch_val in self.batches(*self.datasets["val"], batchSize=1, shuffle=False):
                with torch.set_grad_enabled(False):
                    xBatch_val, yBatch_val = [b.to(device) for b in batch_val]
                    outputs = self.network(xBatch_val[:, 0], xBatch_val[:, 1], self.minMax)
                    loss = criterion(outputs, yBatch_val)
                currentValLoss += loss.item()

            epochTrainLoss = currentTrainLoss / self.dataset_sizes["train"]
            epochValLoss = currentValLoss / self.dataset_sizes["val"]
            stats["val"] = epochValLoss 
            stats["train"] = epochTrainLoss

            if epochValLoss < bestValLoss:
                print("Loss getting better on epoch %d" %(epoch + 1))
                bestValLoss = epochValLoss
                bestWeights = copy.deepcopy(self.network.state_dict())
                noImprovement = 0
            else:
                noImprovement += 1
            
            history.append(stats)
            print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
            if noImprovement >= patience:
                print('early stopping after epoch {epoch:03d}'.format(**stats))
                break

        return bestWeights, history, lrHistory
    
    def modelRMSE(self, device:str):
        """
        Calculates the root mean squared error (RMSE) of the model's predictions on the validation dataset.

        Parameters:
            device (str): The device to use for computation.

        Returns:
            None

        This function iterates over the validation dataset in batches, passing each batch through the network to obtain predictions. It then calculates the RMSE by comparing the ground truth values with the predicted values. The final RMSE is printed to the console.

        Example usage:
        ```
        rmse = modelRMSE(device="cuda")
        ```
        """
        groundTruth, preds = [], []
        for batch in self.batches(*self.datasets["val"], shuffle=False, batchSize=1):
            xBatch, yBatch = [b.to(device) for b in batch]
            outputs = self.network(xBatch[:, 0], xBatch[:, 1], self.minMax)
            groundTruth.extend(yBatch.tolist())
            preds.extend(outputs.tolist())
        groundTruth = np.asarray(groundTruth).ravel()
        preds = np.round(np.asarray(preds).ravel())
        final_loss = np.sqrt(np.mean(preds - groundTruth)**2)
        print(f'Final RMSE: {final_loss:.4f}')

    def getRealUserId(self, userId: int):
        """
        Get the real user ID corresponding to the given user ID.

        Parameters:
            userId (int): The user ID for which to retrieve the real user ID.

        Returns:
            int: The real user ID corresponding to the given user ID.

        Raises:
            IndexError: If no row in the dataframe matches the given user ID.
        """
        realUserId: int = self.df[self.df["newAuthorId"] == userId]["AuthorId"].values[0]
        return realUserId

    def getRealRecipeId(self, recipeId: int):
        """
        Get the real recipe ID corresponding to the given recipe ID.

        Parameters:
            recipeId (int): The recipe ID for which to retrieve the real recipe ID.

        Returns:
            int: The real recipe ID corresponding to the given recipe ID.

        Raises:
            IndexError: If no recipe with the given recipe ID is found.
        """
        realRecipeId: int = self.df[self.df["newRecipeId"] == recipeId]["RecipeId"].values[0]
        return realRecipeId

    def getRecommendation(self, userId: int, topN: int):
        """
        Generates recommendations for a given user ID based on the collaborative filtering model.

        Parameters:
            userId (int): The ID of the user for whom recommendations are generated.
            topN (int): The number of recommendations to generate.

        Returns:
            pandas.DataFrame: A DataFrame containing the top N recommendations for the user,
            with columns "recipeId" (the ID of the recommended recipe), "est" (the estimated rating),
            "userId" (the user ID corresponding to the recommendation), "iid" (the internal recipe ID),
            and "uid" (the user ID corresponding to the recommendation).

        This function first creates a tensor of user IDs by repeating the given user ID for each recipe.
        It then creates a tensor of recipe IDs.
        The function calls the network to predict ratings for the user and recipes, and converts the result to a numpy array.
        The function creates a temporary DataFrame with the predicted ratings and the corresponding recipe IDs.
        It sets the "userId" column of the DataFrame to the given user ID, and applies the "getRealRecipeId" function to the "recipeId" column to get the corresponding internal recipe IDs.
        It also applies the "getRealUserId" function to the "uid" column to get the corresponding user IDs.
        Finally, it sorts the DataFrame by the "est" column in descending order and returns the top N recommendations.
        """
        userIdMultiples = torch.LongTensor(self.newRecipes.shape[0] * [userId])
        tensorRecipes = torch.LongTensor(self.newRecipes)
        pred = self.network(userIdMultiples, tensorRecipes, self.minMax).detach().numpy().reshape(-1)
        dfTemp = pd.DataFrame({"recipeId": self.newRecipes.values, "est": pred})
        dfTemp["userId"] = userIdMultiples
        dfTemp["iid"] = dfTemp["recipeId"].apply(self.getRealRecipeId)
        dfTemp["uid"] = dfTemp["userId"].copy()
        dfTemp["userId"] = dfTemp["userId"].apply(self.getRealUserId)
        return dfTemp.sort_values("est", ascending=False)[:topN]

    def saveModel(self, modelDict, history, lrHistory, path: str):
        """
        Saves the model, history, and learning rate history to the specified path.

        Parameters:
            modelDict (dict): The dictionary containing the model parameters.
            history (list): The list of training and validation loss for each epoch.
            lrHistory (list): The list of learning rates for each epoch.
            path (str): The path to save the model, history, and learning rate history.

        Returns:
            None
        """
        torch.save(modelDict, path)

        with open(f"{path}_history.pkl", "wb") as f:
            pickle.dump(history, f)

        with open(f"{path}_history_lr.pkl", "wb") as f:
            pickle.dump(lrHistory, f)

    def loadModel(self,  path: str):
        """
        Loads a model from the specified path.

        Parameters:
            path (str): The path to the model file.

        Returns:
            torch.nn.Module: The loaded model.
        """
        return torch.load(path)

    def loadPickle(self, path: str, pickleName: str):
        """
        Loads a pickle file from the specified path and returns the loaded object.

        Parameters:
            path (str): The path to the directory where the pickle file is located.
            pickleName (str): The name of the pickle file without the extension.

        Returns:
            The loaded object from the pickle file.
        """
        with open(f"{path}{pickleName}.pkl", "rb") as f:
            return pickle.load(f)