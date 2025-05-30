import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class CO2Predictor:
    def __init__(self, csvPath):
        self.df = pd.read_csv(csvPath)
        self.originalDf = self.df.copy()

    def evaluateModel(self, predictedColumnName):
        validDf = self.df.dropna(subset=["Value", predictedColumnName])

        if len(validDf) == 0:
            return {
                "mse": np.nan,
                "rmse": np.nan,
                "msle": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "r2": np.nan,
                "r": np.nan,
                "r2_adj": np.nan,
                "rse": np.nan,
            }

        actual = validDf["Value"].values
        predicted = validDf[predictedColumnName].values

        errors = actual - predicted
        squaredErr = errors**2
        absErr = np.abs(errors)

        # mse and rmse
        mse = np.mean(squaredErr)
        rmse = np.sqrt(mse)

        # msle (mean squared logarithmic error)
        # only calculate if all values are positive
        if np.all(actual > 0) and np.all(predicted > 0):
            msle = np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)
        else:
            msle = np.nan

        # mae
        mae = np.mean(absErr)

        # mape (mean absolute percentage error)
        nonZeroMask = actual != 0
        if np.any(nonZeroMask):
            mape = np.mean(np.abs(errors[nonZeroMask] / actual[nonZeroMask])) * 100
        else:
            mape = np.nan

        # r² and correlation
        actualMean = np.mean(actual)
        TSS = np.sum((actual - actualMean) ** 2)
        RSS = np.sum(squaredErr)

        if TSS != 0:
            rSquared = 1 - (RSS / TSS)

            actualStd = np.std(actual, ddof=1)
            predictedStd = np.std(predicted, ddof=1)

            if actualStd != 0 and predictedStd != 0:
                covariance = np.mean(
                    (actual - np.mean(actual)) * (predicted - np.mean(predicted))
                )
                r = covariance / (actualStd * predictedStd)
            else:
                r = np.nan

            n = len(actual)

            p = 1
            rSquaredAdj = 1 - ((1 - rSquared) * (n - 1) / (n - p - 1))
            rse = np.sqrt(RSS / (n - 2))
        else:
            rSquared = np.nan
            r = np.nan
            rSquaredAdj = np.nan
            rse = np.nan

        return {
            "mse": mse,
            "rmse": rmse,
            "msle": msle,
            "mae": mae,
            "mape": mape,
            "rSquared": rSquared,
            "r": r,
            "rSquaredAdj": rSquaredAdj,
            "rse": rse,
        }

    def addFutureDates(self, nFuture=12):
        lastDate = pd.to_datetime(self.df["Date"].iloc[-1])

        existingFutureDates = len(self.df[self.df["Value"].isna()])

        datesToAdd = max(0, nFuture - existingFutureDates)

        for i in range(datesToAdd):
            futureDate = lastDate + relativedelta(months=i + 1)
            futureDateStr = futureDate.strftime("%Y-%m-%d")

            newRow = pd.DataFrame({"Date": [futureDateStr], "Value": [np.nan]})
            self.df = pd.concat([self.df, newRow], ignore_index=True)

        self.df = self.df.sort_values("Date").reset_index(drop=True)

    def printResults(self, methodName, columnName, additionalInfo=None):
        print(f"\n{methodName} Result:")
        print("=" * 50)

        if additionalInfo:
            for key, value in additionalInfo.items():
                print(f"{key}: {value}")

        print("\nevaluation (historical data only):")
        evaluation = self.evaluateModel(columnName)
        for metric, value in evaluation.items():
            print(f"  {metric}: {value:.6f}")

        futurePreds = self.df[self.df[columnName].notna() & self.df["Value"].isna()]
        if not futurePreds.empty:
            print("\nfuture predictions:")
            for _, row in futurePreds.iterrows():
                print(f"  {row['Date']}: {row[columnName]:.2f}")

    def movingAvg(self, data, windowSize):
        weights = np.ones(windowSize) / windowSize
        return np.convolve(data, weights, mode="valid")

    def predictWithMovingAvg(self, data, windowSize):
        predictions = []
        workingData = data.copy()

        for _ in range(windowSize):
            nextValue = np.mean(workingData[-windowSize:])
            predictions.append(nextValue)
            workingData = np.append(workingData, nextValue)

        return predictions

    def execMovingAvg(self, windowSize=12):
        self.addFutureDates()

        validData = self.df[self.df["Value"].notna()].copy()
        ma = self.movingAvg(validData["Value"].values, windowSize)

        paddedMa = np.pad(ma, (windowSize - 1, 0), "constant", constant_values=np.nan)
        self.df["MovingAvg"] = np.nan

        validIndices = self.df[self.df["Value"].notna()].index
        for i, idx in enumerate(validIndices):
            if i < len(paddedMa):
                self.df.loc[idx, "MovingAvg"] = paddedMa[i]

        predictions = self.predictWithMovingAvg(validData["Value"].values, windowSize)
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(predictions)]):
            self.df.loc[idx, "MovingAvg"] = predictions[i]

        self.printResults("Moving Average", "MovingAvg", {"window size": windowSize})

    def ratioToMovingAverage(self, data, windowSize):
        ma = self.movingAvg(data, windowSize)
        paddedMa = np.pad(ma, (windowSize - 1, 0), "constant", constant_values=np.nan)
        ratio = data / paddedMa * 100
        return ratio, paddedMa

    def seasonalIndex(self, df, ratioCol):
        dfCopy = df.copy()
        dfCopy["Month"] = pd.to_datetime(dfCopy["Date"]).dt.month
        monthlyIndex = dfCopy.groupby("Month")[ratioCol].mean()
        monthlyIndex = monthlyIndex / monthlyIndex.mean() * 100
        return monthlyIndex

    def execRMA(self, windowSize=12, nFuture=12):
        """execute ratio to moving average prediction"""
        self.addFutureDates(nFuture)

        validDf = self.df[self.df["Value"].notna()].copy()
        data = validDf["Value"].values
        ratio, paddedMa = self.ratioToMovingAverage(data, windowSize)

        if "RMA" not in self.df.columns:
            self.df["RMA"] = np.nan
        if "RMA_Pred" not in self.df.columns:
            self.df["RMA_Pred"] = np.nan

        validIndices = self.df[self.df["Value"].notna()].index
        for i, idx in enumerate(validIndices):
            if i < len(ratio):
                self.df.loc[idx, "RMA"] = ratio[i]
            if i < len(paddedMa):
                self.df.loc[idx, "MovingAvg"] = paddedMa[i]

        idxSeasonal = self.seasonalIndex(validDf.assign(RMA=ratio), "RMA")

        lastDate = pd.to_datetime(validDf.iloc[-1]["Date"])

        for i in range(nFuture):
            futureDate = lastDate + relativedelta(months=i + 1)
            ma = np.mean(data[-windowSize:])
            predMonth = futureDate.month
            seasonal = idxSeasonal.loc[predMonth]
            pred = ma * seasonal / 100

            futureDateStr = futureDate.strftime("%Y-%m-%d")
            mask = self.df["Date"] == futureDateStr
            if mask.any():
                self.df.loc[mask, "RMA_Pred"] = pred

            data = np.append(data, pred)

        for i in range(windowSize - 1, len(validDf)):
            validIdx = validIndices[i]
            month = pd.to_datetime(self.df.loc[validIdx, "Date"]).month
            seasonal = idxSeasonal.loc[month]
            self.df.loc[validIdx, "RMA_Pred"] = (
                self.df.loc[validIdx, "MovingAvg"] * seasonal / 100
            )

        additionalInfo = {
            "window size": windowSize,
            "seasonal index": dict(idxSeasonal.round(3)),
        }
        self.printResults("Ratio to Moving Average", "RMA_Pred", additionalInfo)

    def percentageAverageSeasonalIndex(self, df):
        dfCopy = df.copy()
        dfCopy["Month"] = pd.to_datetime(dfCopy["Date"]).dt.month
        overallAvg = dfCopy["Value"].mean()
        dfCopy["Percentage"] = dfCopy["Value"] / overallAvg * 100
        monthlyIndex = dfCopy.groupby("Month")["Percentage"].mean()
        monthlyIndex = monthlyIndex / monthlyIndex.mean() * 100
        return monthlyIndex, overallAvg

    def execPercentageAverage(self, nFuture=12):
        self.addFutureDates(nFuture)

        validDf = self.df[self.df["Value"].notna()].copy()
        idxSeasonal, overallAvg = self.percentageAverageSeasonalIndex(validDf)

        if "PercAvg_Pred" not in self.df.columns:
            self.df["PercAvg_Pred"] = np.nan

        self.df["Date"] = pd.to_datetime(self.df["Date"])

        historicalRows = self.df[self.df["Value"].notna()]
        for index, row in historicalRows.iterrows():
            month = row["Date"].month
            seasonal = idxSeasonal.loc[month]
            pred = overallAvg * seasonal / 100
            self.df.loc[index, "PercAvg_Pred"] = pred

        futureRows = self.df[self.df["Value"].isna()]
        for i, (index, row) in enumerate(futureRows.iterrows()):
            if i >= nFuture:
                break
            month = row["Date"].month
            seasonal = idxSeasonal.loc[month]
            pred = overallAvg * seasonal / 100
            self.df.loc[index, "PercAvg_Pred"] = pred

        self.df["Date"] = self.df["Date"].dt.strftime("%Y-%m-%d")

        additionalInfo = {
            "overall average": f"{overallAvg:.2f}",
            "seasonal index": dict(idxSeasonal.round(3)),
        }
        self.printResults("Percentage Average", "PercAvg_Pred", additionalInfo)

    def exponentialSmoothing(self, data, alpha):
        result = [data[0]]
        for n in range(1, len(data)):
            result.append(alpha * data[n] + (1 - alpha) * result[n - 1])
        return result

    def predictWithExponentialSmoothing(self, data, alpha, forecastHorizon):
        validData = [x for x in data if pd.notna(x)]
        if len(validData) == 0:
            return [np.nan] * forecastHorizon

        smoothed = self.exponentialSmoothing(validData, alpha)
        lastSmoothedValue = smoothed[-1]

        recentValues = smoothed[-3:] if len(smoothed) >= 3 else smoothed
        if len(recentValues) >= 2:
            trend = (recentValues[-1] - recentValues[0]) / len(recentValues)
        else:
            trend = 0

        predictions = []
        for i in range(forecastHorizon):
            dampingFactor = 0.9**i
            currentValue = lastSmoothedValue + (trend * (i + 1) * dampingFactor)
            predictions.append(currentValue)

        return predictions

    def execExponentialSmoothing(self, alpha=0.3, forecastHorizon=12):
        self.addFutureDates(forecastHorizon)

        actualValues = self.df["Value"].dropna().values
        smoothedValues = self.exponentialSmoothing(actualValues, alpha)

        if "ExpSmooth" not in self.df.columns:
            self.df["ExpSmooth"] = np.nan

        actualIndices = self.df["Value"].dropna().index
        for i, idx in enumerate(actualIndices):
            self.df.loc[idx, "ExpSmooth"] = smoothedValues[i]

        predictions = self.predictWithExponentialSmoothing(
            actualValues, alpha, forecastHorizon
        )
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(predictions)]):
            self.df.loc[idx, "ExpSmooth"] = predictions[i]

        additionalInfo = {"alpha": alpha, "forecast horizon": forecastHorizon}
        self.printResults("Exponential Smoothing", "ExpSmooth", additionalInfo)

    def execMonteCarlo(self, nFuture=12, nSim=1000, windowSize=12):
        self.addFutureDates(nFuture)

        values = self.df["Value"].dropna().values
        changes = values[1:] - values[:-1]
        meanChange = changes.mean()
        stdChange = changes.std()

        if "MonteCarlo_Pred" not in self.df.columns:
            self.df["MonteCarlo_Pred"] = np.nan

        for i in range(windowSize, len(values)):
            sims = []
            for _ in range(nSim):
                simValue = values[i - windowSize]
                for j in range(windowSize):
                    simValue += np.random.normal(meanChange, stdChange)
                sims.append(simValue)
            pred = np.mean(sims)

            valueIndex = self.df["Value"].dropna().index[i]
            self.df.loc[valueIndex, "MonteCarlo_Pred"] = pred

        lastValue = values[-1]
        futurePreds = []

        for i in range(nFuture):
            sims = []
            for _ in range(nSim):
                simValue = lastValue
                for j in range(windowSize + i):
                    simValue += np.random.normal(meanChange, stdChange)
                sims.append(simValue)
            pred = np.mean(sims)
            futurePreds.append(pred)

        futureRows = self.df[self.df["Value"].isna()].index
        for i, idx in enumerate(futureRows[: len(futurePreds)]):
            self.df.loc[idx, "MonteCarlo_Pred"] = futurePreds[i]

        additionalInfo = {
            "simulations": nSim,
            "window size": windowSize,
            "mean change": f"{meanChange:.6f}",
            "std change": f"{stdChange:.6f}",
        }
        self.printResults("Monte Carlo Simulation", "MonteCarlo_Pred", additionalInfo)

    def findOptimalArima(self, timeSeries, maxP=5, maxD=2, maxQ=5):
        bestAic = np.inf
        bestOrder = None

        print("finding optimal ARIMA parameters...")

        for p in range(maxP + 1):
            for d in range(maxD + 1):
                for q in range(maxQ + 1):
                    try:
                        model = ARIMA(timeSeries, order=(p, d, q))
                        modelFit = model.fit()

                        if modelFit.aic < bestAic:
                            bestAic = modelFit.aic
                            bestOrder = (p, d, q)

                    # i will use it bare, fk u
                    except:
                        continue

        print(f"best ARIMA order: {bestOrder}")
        print(f"best AIC: {bestAic}")
        return bestOrder

    def execArima(self, nFuture=12, order=None):
        self.addFutureDates(nFuture)

        dfCopy = self.df.copy()
        dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
        dfCopy.set_index("Date", inplace=True)

        timeSeries = dfCopy["Value"].dropna()

        # there should be no need of ever using these? i think? idk imma put this shi hir
        if order is None:
            order = self.findOptimalArima(timeSeries)

        model = ARIMA(timeSeries, order=order)
        modelFit = model.fit()

        # yes it print out sarima, i think its because of the overlaying model that do so.
        # but in the end, as long as there is no seasonal order, it technically is ARIMA
        # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
        # cite?, integration models: ARIMA(p, d, q)
        print(f"\nARIMA{order} Model Summary:")
        print(f"AIC: {modelFit.aic:.4f}")
        print(f"BIC: {modelFit.bic:.4f}")
        print(f"Log Likelihood: {modelFit.llf:.4f}")

        inSamplePred = modelFit.fittedvalues

        if "ArimaPred" not in self.df.columns:
            self.df["ArimaPred"] = np.nan

        validIndices = self.df["Value"].dropna().index
        for i, idx in enumerate(validIndices):
            if i < len(inSamplePred):
                if i > 0:
                    self.df.loc[idx, "ArimaPred"] = inSamplePred.iloc[i]

        # make future predictions
        forecast = modelFit.forecast(steps=nFuture)
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(forecast)]):
            self.df.loc[idx, "ArimaPred"] = (
                forecast.iloc[i] if hasattr(forecast, "iloc") else forecast[i]
            )

        additionalInfo = {
            "order": f"({order[0]}, {order[1]}, {order[2]})",
            "AIC": f"{modelFit.aic:.4f}",
            "BIC": f"{modelFit.bic:.4f}",
        }
        self.printResults("ARIMA", "ArimaPred", additionalInfo)

    def findOptimalSarima(
        self,
        timeSeries,
        maxP=3,
        maxD=2,
        maxQ=3,
        maxSeasonalP=2,
        maxSeasonalD=1,
        maxSeasonalQ=2,
        seasonalPeriod=12,
    ):
        bestAic = np.inf
        bestOrder = None
        bestSeasonalOrder = None

        print("Finding optimal SARIMA parameters...")

        for p in range(maxP + 1):
            for d in range(maxD + 1):
                for q in range(maxQ + 1):
                    for P in range(maxSeasonalP + 1):
                        for D in range(maxSeasonalD + 1):
                            for Q in range(maxSeasonalQ + 1):
                                try:
                                    model = SARIMAX(
                                        timeSeries,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonalPeriod),
                                    )
                                    modelFit = model.fit(disp=False)

                                    if modelFit.aic < bestAic:
                                        bestAic = modelFit.aic
                                        bestOrder = (p, d, q)
                                        bestSeasonalOrder = (P, D, Q, seasonalPeriod)

                                except:
                                    continue

        print(f"Best SARIMA order: {bestOrder}")
        print(f"Best seasonal order: {bestSeasonalOrder}")
        print(f"Best AIC: {bestAic}")
        return bestOrder, bestSeasonalOrder

    def execSarima(
        self,
        nFuture=12,
        order=(1, 1, 1),
        seasonalOrder=(1, 1, 1, 12),
    ):
        self.addFutureDates(nFuture)  # add missing line
        dfCopy = self.df.copy()
        dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
        dfCopy.set_index("Date", inplace=True)

        timeSeries = dfCopy["Value"].dropna()

        if order is None:
            order, seasonalOrder = self.findOptimalSarima(timeSeries)

        model = SARIMAX(timeSeries, order=order, seasonal_order=seasonalOrder)
        modelFit = model.fit()

        print(f"\nSARIMA{order} Model Summary:")
        print(f"AIC: {modelFit.aic:.4f}")
        print(f"BIC: {modelFit.bic:.4f}")
        print(f"Log Likelihood: {modelFit.llf:.4f}")

        inSamplePred = modelFit.fittedvalues
        if "SarimaPred" not in self.df.columns:
            self.df["SarimaPred"] = np.nan

        validIndices = self.df["Value"].dropna().index
        for i, idx in enumerate(validIndices):
            if i < len(inSamplePred):
                if i >= 12:
                    self.df.loc[idx, "SarimaPred"] = inSamplePred.iloc[i]

        # make future predictions
        forecast = modelFit.forecast(steps=nFuture)
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(forecast)]):
            self.df.loc[idx, "SarimaPred"] = (
                forecast.iloc[i] if hasattr(forecast, "iloc") else forecast[i]
            )

        additionalInfo = {
            "order": f"({order[0]}, {order[1]}, {order[2]})",
            "AIC": f"{modelFit.aic:.4f}",
            "BIC": f"{modelFit.bic:.4f}",
        }

        self.printResults("SARIMA", "SarimaPred", additionalInfo)

    def execLinearRegression(self, steps=12):
        self.addFutureDates(steps)

        dfCopy = self.df.copy()
        dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])

        startDate = dfCopy["Date"].min()
        dfCopy["timeIndex"] = (dfCopy["Date"] - startDate).dt.days / 30.44

        validData = dfCopy.dropna(subset=["Value"])
        x = validData["timeIndex"].values.reshape(-1, 1)
        y = validData["Value"].values

        model = LinearRegression()
        model.fit(x, y)

        print("\nLinear Regression Model:")
        print(f"R² Score: {model.score(x, y):.6f}")
        print(f"Coefficient: {model.coef_[0]:.6f}")
        print(f"Intercept: {model.intercept_:.6f}")

        if "LinearRegPred" not in self.df.columns:
            self.df["LinearRegPred"] = np.nan

        allTimeIndex = (pd.to_datetime(self.df["Date"]) - startDate).dt.days / 30.44
        allPredictions = model.predict(allTimeIndex.values.reshape(-1, 1))

        for i, idx in enumerate(self.df.index):
            self.df.loc[idx, "LinearRegPred"] = allPredictions[i]

        additionalInfo = {
            "R² score": f"{model.score(x, y):.6f}",
            "coefficient": f"{model.coef_[0]:.6f}",
            "intercept": f"{model.intercept_:.6f}",
        }
        self.printResults("Linear Regression", "LinearRegPred", additionalInfo)

    def plotTimeSeries(
        self, valueCol="Value", otherCols=None, title="CO2 Levels Over Time"
    ):
        if otherCols is None:
            otherCols = []

        plt.figure(figsize=(14, 6))

        dates = pd.to_datetime(self.df["Date"])

        plt.plot(
            dates,
            self.df[valueCol],
            color="royalblue",
            linewidth=2,
            label=f"Actual {valueCol}",
        )

        colors = sns.color_palette("Set2", len(otherCols))
        for i, col in enumerate(otherCols):
            if col in self.df.columns:
                plt.plot(
                    dates,
                    self.df[col],
                    color=colors[i],
                    linewidth=2,
                    linestyle="--",
                    label=f"{col}",
                )

        lastActualDate = dates[~self.df[valueCol].isna()].max()
        plt.axvline(x=lastActualDate, color="gray", linestyle="--", alpha=0.7)
        plt.text(
            lastActualDate,
            self.df[valueCol].max(),
            " Predictions →",
            verticalalignment="top",
        )

        plt.title(title, fontsize=14)
        plt.ylabel("CO2 (ppm)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        return plt

    def plotYearlyAvg(
        self, valueCol="Value", otherCols=None, title="Annual CO2 Levels"
    ):
        if otherCols is None:
            otherCols = []

        dfCopy = self.df.copy()
        dfCopy["Year"] = pd.to_datetime(dfCopy["Date"]).dt.year

        aggDict = {valueCol: "mean"}
        for col in otherCols:
            if col in dfCopy.columns:
                aggDict[col] = "mean"

        yearlyData = dfCopy.groupby("Year").agg(aggDict).reset_index()
        lastActualYear = dfCopy[~dfCopy[valueCol].isna()]["Year"].max()

        plt.figure(figsize=(14, 6))

        plt.plot(
            yearlyData["Year"],
            yearlyData[valueCol],
            marker="o",
            color="royalblue",
            linewidth=2,
            label=f"Actual {valueCol}",
        )

        colors = sns.color_palette("Set2", len(otherCols))
        for i, col in enumerate(otherCols):
            if col in yearlyData.columns:
                plt.plot(
                    yearlyData["Year"],
                    yearlyData[col],
                    marker="x",
                    linestyle="--",
                    color=colors[i],
                    linewidth=2,
                    label=f"{col}",
                )

        plt.axvline(x=lastActualYear, color="gray", linestyle="--", alpha=0.7)
        plt.text(
            lastActualYear,
            yearlyData[valueCol].max(),
            " Predictions →",
            verticalalignment="top",
        )

        plt.title(title, fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Average CO2 (ppm)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        return plt

    def runAllMethods(self):
        """run all prediction methods"""
        print("CO2 Prediction Analysis")
        print("=" * 60)

        self.execMovingAvg(windowSize=12)
        self.execRMA(windowSize=12, nFuture=12)
        self.execPercentageAverage(nFuture=12)
        self.execExponentialSmoothing(alpha=0.3, forecastHorizon=12)
        self.execMonteCarlo(nFuture=12, nSim=1000, windowSize=12)
        self.execArima(nFuture=12, order=(5, 1, 5))
        self.execSarima(nFuture=12, order=(2, 0, 1), seasonalOrder=(0, 1, 2, 12))
        self.execLinearRegression(steps=12)

        predictionCols = [
            "MovingAvg",
            "RMA_Pred",
            "PercAvg_Pred",
            "ExpSmooth",
            "MonteCarlo_Pred",
            "ArimaPred",
            "SarimaPred",
            "LinearRegPred",
        ]

        self.plotTimeSeries(
            otherCols=predictionCols, title="CO2 Growth with All Prediction Methods"
        )
        plt.savefig(
            os.path.join(dir, "../plots/all_predictions_timeseries.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        self.plotYearlyAvg(
            otherCols=predictionCols, title="Average Annual CO2 Levels (All Methods)"
        )
        plt.savefig(
            os.path.join(dir, "../plots/all_predictions_yearly.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        return self.df

    def exportResults(self, outputPath="../processed-data2.csv"):
        """export results to CSV"""
        self.df.to_csv(outputPath, index=False)
        print(f"\nresults exported to: {outputPath}")


if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    predictor = CO2Predictor(os.path.join(dir, "..", "preprocessed-data.csv"))

    resultDf = predictor.runAllMethods()

    predictor.exportResults(outputPath=os.path.join(dir, "..", "processed-data.csv"))

    print("\nFinal DataFrame Preview:")
    print("=" * 60)
    print(resultDf.head(20))
    print(f"\nTotal rows: {len(resultDf)}")
    print(f"Columns: {list(resultDf.columns)}")
