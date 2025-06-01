import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

warnings.filterwarnings("ignore")


def createFeatures(timeSeries, lookBack=12):
    features = []
    targets = []

    for i in range(lookBack, len(timeSeries)):
        features.append(timeSeries[i - lookBack : i])
        targets.append(timeSeries[i])

    return np.array(features), np.array(targets)


def optimizeSvrParameters(
    xTrain,
    yTrain,
    cValues=[0.01, 0.1, 1, 5, 10, 50, 75, 100, 250, 500, 750, 1000, 2000, 5000],
    gammaValues=["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1, 10, 50],
    epsilonValues=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5],
    cvFolds=5,
    nJobs=-1,
    scoring="neg_mean_squared_error",
):
    print("setting up SVR parameter grid...")

    # we want non linear, thus rbf,
    pipeline = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])

    paramGrid = {
        "svr__C": cValues,
        "svr__gamma": gammaValues,
        "svr__epsilon": epsilonValues,
    }

    tscv = TimeSeriesSplit(n_splits=cvFolds)

    print(
        f"starting gridsearch with {len(cValues) * len(gammaValues) * len(epsilonValues)} combinations..."
    )

    gridSearch = GridSearchCV(
        pipeline,
        paramGrid,
        cv=tscv,
        scoring=scoring,
        n_jobs=nJobs,
        verbose=1,
        return_train_score=True,
    )

    # fit the gridsearch
    gridSearch.fit(xTrain, yTrain)

    return gridSearch


def evaluateModel(model, xTest, yTest):
    predictions = model.predict(xTest)

    mse = mean_squared_error(yTest, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yTest, predictions)
    r2 = r2_score(yTest, predictions)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "predictions": predictions}


def getBestSvrParams(csvPath="preprocessed-data.csv", lookBack=12, testSize=0.2):
    print("loading and preparing data...")

    df = pd.read_csv(csvPath)
    dfCopy = df.copy()
    dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
    dfCopy.set_index("Date", inplace=True)

    timeSeries = dfCopy["Value"].values

    features, targets = createFeatures(timeSeries, lookBack)

    splitIndex = int(len(features) * (1 - testSize))
    xTrain, xTest = features[:splitIndex], features[splitIndex:]
    yTrain, yTest = targets[:splitIndex], targets[splitIndex:]

    print(f"training samples: {len(xTrain)}")
    print(f"test samples: {len(xTest)}")
    print(f"lookback window: {lookBack}")

    gridSearch = optimizeSvrParameters(xTrain, yTrain)

    bestModel = gridSearch.best_estimator_
    bestParams = gridSearch.best_params_
    bestScore = gridSearch.best_score_

    print(f"\nbest parameters: {bestParams}")
    print(f"best cross-validation score: {bestScore:.4f}")

    testResults = evaluateModel(bestModel, xTest, yTest)

    print("\ntest set performance:")
    print(f"RMSE: {testResults['rmse']:.4f}")
    print(f"MAE: {testResults['mae']:.4f}")
    print(f"R²: {testResults['r2']:.4f}")

    return bestModel, bestParams, testResults


if __name__ == "__main__":
    print("SVR hyperparameter optimization starting...\n")

    # optimize SVR parameters
    bestModel, bestParams, testResults = getBestSvrParams()

    print("\noptimization completed!")
    print(f"final best parameters: {bestParams}")
