import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")


def evaluateSarimaModel(params, timeSeries, seasonalPeriod):
    try:
        p, d, q, P, D, Q = params
        model = SARIMAX(
            timeSeries,
            order=(p, d, q),
            seasonal_order=(P, D, Q, seasonalPeriod),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        modelFit = model.fit(disp=False, maxiter=50)
        return modelFit.aic, (p, d, q), (P, D, Q, seasonalPeriod)
    except:
        return np.inf, None, None


def generateParamCombinations(
    maxP, maxD, maxQ, maxSeasonalP, maxSeasonalD, maxSeasonalQ
):
    combinations = []
    for p in range(maxP + 1):
        for d in range(maxD + 1):
            for q in range(maxQ + 1):
                for P in range(maxSeasonalP + 1):
                    for D in range(maxSeasonalD + 1):
                        for Q in range(maxSeasonalQ + 1):
                            combinations.append((p, d, q, P, D, Q))
    return combinations


def findOptimalSarima(
    timeSeries,
    maxP=4,
    maxD=4,
    maxQ=4,
    maxSeasonalP=4,
    maxSeasonalD=4,
    maxSeasonalQ=4,
    seasonalPeriod=12,
    numProcesses=None,
):
    if numProcesses is None:
        numProcesses = 6

    print(f"finding optimal SARIMA parameters using {numProcesses} processes...")

    paramCombinations = generateParamCombinations(
        maxP, maxD, maxQ, maxSeasonalP, maxSeasonalD, maxSeasonalQ
    )
    totalCombinations = len(paramCombinations)

    evaluateFunc = partial(
        evaluateSarimaModel, timeSeries=timeSeries, seasonalPeriod=seasonalPeriod
    )

    results = []
    with Pool(processes=numProcesses) as pool:
        with tqdm(total=totalCombinations, desc="evaluating models") as pbar:
            for result in pool.imap(evaluateFunc, paramCombinations):
                results.append(result)
                pbar.update(1)

    validResults = [
        (aic, order, seasonalOrder)
        for aic, order, seasonalOrder in results
        if order is not None
    ]

    if not validResults:
        print("no valid SARIMA model found!")
        return None, None

    validResults.sort(key=lambda x: x[0])
    bestAic, bestOrder, bestSeasonalOrder = validResults[0]

    print(f"\nbest SARIMA order: {bestOrder}")
    print(f"best seasonal order: {bestSeasonalOrder}")
    print(f"best AIC: {bestAic:.4f}")

    print("\ntop 5 best models:")
    for i, (aic, order, seasonalOrder) in enumerate(validResults[:5]):
        print(f"{i + 1}. order={order}, seasonal={seasonalOrder}, AIC={aic:.4f}")

    return bestOrder, bestSeasonalOrder


def getBestSarimaParams(csvPath="preprocessed-data.csv"):
    df = pd.read_csv(csvPath)

    dfCopy = df.copy()
    dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
    dfCopy.set_index("Date", inplace=True)

    timeSeries = dfCopy["Value"].dropna()

    print(f"using {len(timeSeries)} data points for parameter optimization\n")

    bestOrder, bestSeasonalOrder = findOptimalSarima(timeSeries)

    return bestOrder, bestSeasonalOrder


if __name__ == "__main__":
    bestOrder, bestSeasonalOrder = getBestSarimaParams()

    print("\nfinal results:")
    print(f"best SARIMA order: {bestOrder}")
    print(f"best seasonal order: {bestSeasonalOrder}")
