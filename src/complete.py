import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
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

        self.df["MovingAvg"] = np.nan

        for i in range(windowSize, len(validData) - 1):
            previousValues = validData["Value"].iloc[i - windowSize : i].values
            movingAvg = np.mean(previousValues)

            validIdx = validData.index[i]
            self.df.loc[validIdx, "MovingAvg"] = movingAvg

        lastWindowValues = validData["Value"].iloc[-windowSize:].values
        predictions = self.predictWithMovingAvg(lastWindowValues, windowSize)
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(predictions)]):
            self.df.loc[idx, "MovingAvg"] = predictions[i]

        print("df", self.df.to_string())

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
        self.addFutureDates(nFuture)

        validDf = self.df[self.df["Value"].notna()].copy()
        data = validDf["Value"].values
        ratio, paddedMa = self.ratioToMovingAverage(data, windowSize)

        if "RMA" not in self.df.columns:
            self.df["RMA"] = np.nan
        if "RMA_Pred" not in self.df.columns:
            self.df["RMA_Pred"] = np.nan

        validIndices = self.df[self.df["Value"].notna()].index

        # start rma from windowSize position like movingAvg
        for i, idx in enumerate(validIndices):
            if i >= windowSize and i < len(ratio):
                self.df.loc[idx, "RMA"] = ratio[i]

        seasonalIdx = self.seasonalIndex(validDf.assign(RMA=ratio), "RMA")

        for i in range(windowSize, len(validDf)):
            if i < len(validIndices):
                validIdx = validIndices[i]
                month = pd.to_datetime(self.df.loc[validIdx, "Date"]).month
                seasonal = seasonalIdx.loc[month]

                if not pd.isna(self.df.loc[validIdx, "MovingAvg"]):
                    movingAvgValue = self.df.loc[validIdx, "MovingAvg"]
                else:
                    movingAvgValue = (
                        paddedMa[i]
                        if i < len(paddedMa)
                        else np.mean(data[i - windowSize : i])
                    )

                self.df.loc[validIdx, "RMA_Pred"] = movingAvgValue * seasonal / 100

        lastDate = pd.to_datetime(validDf.iloc[-1]["Date"])

        for i in range(nFuture):
            futureDate = lastDate + relativedelta(months=i + 1)
            ma = np.mean(data[-windowSize:])
            predMonth = futureDate.month
            seasonal = seasonalIdx.loc[predMonth]
            pred = ma * seasonal / 100

            futureDateStr = futureDate.strftime("%Y-%m-%d")
            mask = self.df["Date"] == futureDateStr
            if mask.any():
                self.df.loc[mask, "RMA_Pred"] = pred

            data = np.append(data, pred)

        additionalInfo = {
            "window size": windowSize,
            "seasonal index": dict(seasonalIdx.round(3)),
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

    def execArima(self, colName, order, nFuture=12):
        self.addFutureDates(nFuture)

        dfCopy = self.df.copy()
        dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
        dfCopy.set_index("Date", inplace=True)

        timeSeries = dfCopy["Value"].dropna()

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

        if colName not in self.df.columns:
            self.df[colName] = np.nan

        validIndices = self.df["Value"].dropna().index
        for i, idx in enumerate(validIndices):
            if i < len(inSamplePred):
                if i > 0:
                    self.df.loc[idx, colName] = inSamplePred.iloc[i]

        # make future predictions
        forecast = modelFit.forecast(steps=nFuture)
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(forecast)]):
            self.df.loc[idx, colName] = (
                forecast.iloc[i] if hasattr(forecast, "iloc") else forecast[i]
            )

        additionalInfo = {
            "order": f"({order[0]}, {order[1]}, {order[2]})",
            "AIC": f"{modelFit.aic:.4f}",
            "BIC": f"{modelFit.bic:.4f}",
        }
        self.printResults("ARIMA", colName, additionalInfo)

    def execSarima(
        self,
        colName,
        nFuture=12,
        order=(1, 1, 1),
        seasonalOrder=(1, 1, 1, 12),
    ):
        self.addFutureDates(nFuture) 
        dfCopy = self.df.copy()
        dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])
        dfCopy.set_index("Date", inplace=True)

        timeSeries = dfCopy["Value"].dropna()

        model = SARIMAX(timeSeries, order=order, seasonal_order=seasonalOrder)
        modelFit = model.fit()

        print(f"\nSARIMA{order} Model Summary:")
        print(f"AIC: {modelFit.aic:.4f}")
        print(f"BIC: {modelFit.bic:.4f}")
        print(f"Log Likelihood: {modelFit.llf:.4f}")

        inSamplePred = modelFit.fittedvalues
        if colName not in self.df.columns:
            self.df[colName] = np.nan

        validIndices = self.df["Value"].dropna().index
        for i, idx in enumerate(validIndices):
            if i < len(inSamplePred):
                if i >= 12:
                    self.df.loc[idx, colName] = inSamplePred.iloc[i]

        # make future predictions
        forecast = modelFit.forecast(steps=nFuture)
        futureRows = self.df[self.df["Value"].isna()].index

        for i, idx in enumerate(futureRows[: len(forecast)]):
            self.df.loc[idx, colName] = (
                forecast.iloc[i] if hasattr(forecast, "iloc") else forecast[i]
            )

        additionalInfo = {
            "order": f"({order[0]}, {order[1]}, {order[2]})",
            "AIC": f"{modelFit.aic:.4f}",
            "BIC": f"{modelFit.bic:.4f}",
        }

        self.printResults("SARIMA", colName, additionalInfo)

    def execLinearRegression(self, colName, steps=12):
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

        if colName not in self.df.columns:
            self.df[colName] = np.nan

        allTimeIndex = (pd.to_datetime(self.df["Date"]) - startDate).dt.days / 30.44
        allPredictions = model.predict(allTimeIndex.values.reshape(-1, 1))

        for i, idx in enumerate(self.df.index):
            self.df.loc[idx, colName] = allPredictions[i]

        additionalInfo = {
            "R² score": f"{model.score(x, y):.6f}",
            "coefficient": f"{model.coef_[0]:.6f}",
            "intercept": f"{model.intercept_:.6f}",
        }
        self.printResults("Linear Regression", colName, additionalInfo)

    def execSVR(
        self,
        colName,
        steps=12,
        kernel="linear",
        C=0.1,
        gamma="auto",
        epsilon=1.0,
    ):
        self.addFutureDates(steps)

        dfCopy = self.df.copy()
        dfCopy["Date"] = pd.to_datetime(dfCopy["Date"])

        startDate = dfCopy["Date"].min()
        dfCopy["timeIndex"] = (dfCopy["Date"] - startDate).dt.days / 30.44

        validData = dfCopy.dropna(subset=["Value"])
        x = validData["timeIndex"].values.reshape(-1, 1)
        y = validData["Value"].values

        model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        model.fit(x, y)

        if colName not in self.df.columns:
            self.df[colName] = np.nan

        allTimeIndex = (pd.to_datetime(self.df["Date"]) - startDate).dt.days / 30.44
        allPredictions = model.predict(allTimeIndex.values.reshape(-1, 1))

        for i, idx in enumerate(self.df.index):
            self.df.loc[idx, colName] = allPredictions[i]

        additionalInfo = {
            "R² score": f"{model.score(x, y):.6f}",
        }
        self.printResults("SVR", colName, additionalInfo)

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
        print("CO2 Prediction Analysis")
        print("=" * 60)

        self.execMovingAvg(windowSize=12)
        self.execRMA(windowSize=12, nFuture=12)
        self.execPercentageAverage(nFuture=12)
        self.execExponentialSmoothing(alpha=0.3, forecastHorizon=12)
        self.execMonteCarlo(nFuture=12, nSim=1000, windowSize=12)
        self.execArima(
            nFuture=12,
            order=(5, 1, 5),
            colName="ARIMA-OPT",
        )
        self.execSarima(
            nFuture=12,
            order=(2, 0, 1),
            seasonalOrder=(0, 1, 2, 12),
            colName="SARIMA-OPT",
        )

        self.execArima(
            nFuture=12,
            order=(1, 1, 1),
            colName="ARIMA-UNOPT",
        )
        self.execSarima(
            nFuture=12,
            order=(1, 1, 1),
            seasonalOrder=(1, 1, 1, 12),
            colName="SARIMA-UNOPT",
        )

        self.execLinearRegression(colName="LINREG-UNOPT", steps=12)
        self.execSVR(
            colName="SVR-UNOPT",
            steps=12,
            epsilon=1.0,
            kernel="rbf",
            C=0.1,
            gamma="auto",
        )

        self.execSVR(
            colName="SVR-OPT",
            steps=12,
            epsilon=0.1,
            kernel="rbf",
            C=1000,
            gamma=0.0001,
        )

        predictionCols = [
            "MovingAvg",
            "RMA_Pred",
            "PercAvg_Pred",
            "ExpSmooth",
            "MonteCarlo_Pred",
            "ARIMA-OPT",
            "SARIMA-OPT",
            "ARIMA-UNOPT",
            "SARIMA-UNOPT",
            "LINREG-UNOPT",
            "SVR-UNOPT",
            "SVR-OPT",
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
