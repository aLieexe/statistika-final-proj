import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta


class CO2Predictor:
    def __init__(self, csvPath):
        self.df = pd.read_csv(csvPath)
        self.originalDf = self.df.copy()

    def evaluateModel(self, predictedColumnName):
        """evaluate model performance using MSE, RMSE, MAE, and MAPE"""
        validDf = self.df.dropna(subset=["Value", predictedColumnName])

        if len(validDf) == 0:
            return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}

        actual = validDf["Value"]
        predicted = validDf[predictedColumnName]

        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))

        nonZeroActual = actual[actual != 0]
        nonZeroPredicted = predicted[actual != 0]

        if len(nonZeroActual) > 0:
            mape = (
                np.mean(np.abs((nonZeroActual - nonZeroPredicted) / nonZeroActual))
                * 100
            )
        else:
            mape = np.nan

        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

    def addFutureDates(self, nFuture=12):
        lastDate = pd.to_datetime(self.df["Date"].iloc[-1])

        for i in range(nFuture):
            futureDate = lastDate + relativedelta(months=i + 1)
            futureDateStr = futureDate.strftime("%Y-%m-%d")

            if futureDateStr not in self.df["Date"].values:
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
        """calculate seasonal index from ratios"""
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
        """calculate seasonal index using percentage average method"""
        dfCopy = df.copy()
        dfCopy["Month"] = pd.to_datetime(dfCopy["Date"]).dt.month
        overallAvg = dfCopy["Value"].mean()
        dfCopy["Percentage"] = dfCopy["Value"] / overallAvg * 100
        monthlyIndex = dfCopy.groupby("Month")["Percentage"].mean()
        monthlyIndex = monthlyIndex / monthlyIndex.mean() * 100
        return monthlyIndex, overallAvg

    def execPercentageAverage(self, nFuture=12):
        """execute percentage average prediction"""
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
        """calculate exponential smoothing values"""
        result = [data[0]]
        for n in range(1, len(data)):
            result.append(alpha * data[n] + (1 - alpha) * result[n - 1])
        return result

    def predictWithExponentialSmoothing(self, data, alpha, forecastHorizon):
        """predict using exponential smoothing with trend"""
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
        """execute exponential smoothing prediction"""
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
        """execute monte carlo simulation prediction"""
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

    def plotTimeSeries(
        self, valueCol="Value", otherCols=None, title="CO2 Levels Over Time"
    ):
        """plot time series data with predictions"""
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
        """plot yearly average CO2 levels"""
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

        predictionCols = [
            "MovingAvg",
            "RMA_Pred",
            "PercAvg_Pred",
            "ExpSmooth",
            "MonteCarlo_Pred",
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
