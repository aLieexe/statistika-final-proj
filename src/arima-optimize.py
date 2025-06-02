import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

warnings.filterwarnings("ignore")


def evaluateArimaModel(params, timeSeries):
    try:
        p, d, q = params
        model = ARIMA(
            timeSeries,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        modelFit = model.fit()
        return modelFit.aic, (p, d, q)
    except:
        return np.inf, None


def generateParamCombinations(maxP, maxD, maxQ):
    combinations = []
    for p in range(maxP + 1):
        for d in range(maxD + 1):
            for q in range(maxQ + 1):
                combinations.append((p, d, q))
    return combinations


def findOptimalArima(
    timeSeries,
    maxP=5,
    maxD=5,
    maxQ=5,
    numProcesses=None,
):
    if numProcesses is None:
        numProcesses = 6

    print(f"finding optimal ARIMA parameters using {numProcesses} processes...")

    paramCombinations = generateParamCombinations(maxP, maxD, maxQ)
    totalCombinations = len(paramCombinations)

    evaluateFunc = partial(evaluateArimaModel, timeSeries=timeSeries)

    results = []
    with Pool(processes=numProcesses) as pool:
        with tqdm(total=totalCombinations, desc="evaluating models") as pbar:
            for result in pool.imap(evaluateFunc, paramCombinations):
                results.append(result)
                pbar.update(1)

    validResults = [(aic, order) for aic, order in results if order is not None]

    if not validResults:
        print("no valid ARIMA model found!")
        return None

    validResults.sort(key=lambda x: x[0])
    bestAic, bestOrder = validResults[0]

    print(f"\nbest ARIMA order: {bestOrder}")
    print(f"best AIC: {bestAic:.4f}")

    print("\ntop 5 best models:")
    for i, (aic, order) in enumerate(validResults[:5]):
        print(f"{i + 1}. order={order}, AIC={aic:.4f}")

    return bestOrder


def getBestArimaParams(csvPath="preprocessed-data.csv"):
    df = pd.read_csv(csvPath)

    dfCopy = df.copy()
    dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
    dfCopy.set_index("Date", inplace=True)

    timeSeries = dfCopy["Value"].dropna()

    print(f"using {len(timeSeries)} data points for parameter optimization\n")

    bestOrder = findOptimalArima(timeSeries)

    return bestOrder


if __name__ == "__main__":
    bestOrder = getBestArimaParams()

    print("\nfinal results:")
    print(f"best ARIMA order: {bestOrder}")
