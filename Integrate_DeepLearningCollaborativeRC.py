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
        return self
    
    def __next__(self):
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
        features = torch.cat([self.users(users), self.recipes(recipes)], dim=1)
        x = self.embedDropout(features)
        x = self.hiddenLayers(x)
        out = torch.sigmoid(self.output(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out * (max_rating - min_rating + 1) + min_rating - 0.5
        return out
        
    
    def layerGenerator(self, n_in):
        for n_out, drop_rate in zip_longest(self.hidden, self.dropouts):
            yield nn.Linear(n_in, n_out)
            yield nn.ReLU()
            if drop_rate is not None and drop_rate > 0:
                yield nn.Dropout(drop_rate)
            n_in = n_out

    def makeList(self, n):
        if isinstance(n, (int, float)):
            return [n]
        elif hasattr(n, "__iter__"):
            return list(n)
        raise TypeError("Layers configuration must be a single number or a list of numbers")

    def initWeight(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)

    def _init(self):
        self.users.weight.data.uniform_(-0.05, 0.05)
        self.recipes.weight.data.uniform_(-0.05, 0.05)
        self.hiddenLayers.apply(self.initWeight)
        self.initWeight(self.output)


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
    
class DeepLearningCollaborativeBasedRecommender():
    def __init__(self, datasetPath: str, useCols, seed: int = 0):
        self.setupSeed(seed)
        self.df = pd.read_csv(datasetPath, usecols=useCols)
        self.minMax = self.df["Rating"].min().astype(float), self.df["Rating"].max().astype(float)
        self.network: EmbeddingNetwork = None

    def setupSeed(self, seed: int):
        listOfSeeds : List = [np.random.seed, torch.manual_seed, torch.cuda.manual_seed]
        for l in listOfSeeds:
            l(seed)

    def alignDataset(self, target: List[str]):
        self.df = self.df[target]

    def createDataset(self):
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
        for xb, yb in RecipesIterator(X,
                                      y,
                                      batchSize, shuffle):
            xb : torch.Tensor = torch.LongTensor(xb)
            yb : torch.Tensor = torch.FloatTensor(yb)
            yield xb, yb.view(-1, 1)

    def cosineAnnealing(self, num_epochs_per_cycle, min_lr: float=0):
        
        def scheduler(epoch, base_lr: float):
            currentEpochWithinCycle = epoch % num_epochs_per_cycle 
            adjustedLr = min_lr + (base_lr - min_lr) * ( 1 + math.cos(math.pi * currentEpochWithinCycle / num_epochs_per_cycle)) / 2
            return adjustedLr

        return scheduler

    def splitDataset(self, testSize: float = 0.1):
        X = pd.DataFrame({"AuthorId": self.df["newAuthorId"], "RecipeId": self.df["newRecipeId"]})
        Y = self.df["Rating"].astype(float)
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=testSize, random_state=0)
        self.datasets = {"train": (x_train, y_train), "val": (x_val, y_val)}
        self.dataset_sizes = {"train": len(x_train), "val": len(x_val)}

    def setupNetwork(self, nFactors, hiddenLayers= [500, 500, 500],
                     embeddingDropout = 0.05, dropouts=[0.50, 0.50, 0.35]):
        self.network = EmbeddingNetwork(self.nUsers, self.nRecipes, nFactors,
                                        embeddingDropout, hiddenLayers, dropouts)
    
    def trainNetwork(self, epochs: int = 1000,
                    batchSize: int = 2000, device: str="cpu",
                    lr: float=1e-3, weight_decay: float=0.08, patience: int= 15):
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
        realUserId: int = self.df[self.df["newAuthorId"] == userId]["AuthorId"].values[0]
        return realUserId

    def getRealRecipeId(self, recipeId: int):
        realRecipeId: int = self.df[self.df["newRecipeId"] == recipeId]["RecipeId"].values[0]
        return realRecipeId

    def getRecommendation(self, userId: int, topN: int):
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
        torch.save(modelDict, path)

        with open(f"{path}_history.pkl", "wb") as f:
            pickle.dump(history, f)

        with open(f"{path}_history_lr.pkl", "wb") as f:
            pickle.dump(lrHistory, f)

    def loadModel(self,  path: str):
        return torch.load(path)

    def loadPickle(self, path: str, pickleName: str):
        with open(f"{path}{pickleName}.pkl", "rb") as f:
            return pickle.load(f)