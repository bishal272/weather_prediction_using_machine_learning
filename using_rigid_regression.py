# library that supports dataframes  https://www.youtube.com/watch?v=baqxBO4PhI8
import pandas as pd
import matplotlib.pyplot as plt

# machine_learning_modules-->
# this module contains many utilities that will help us choose between models.
# for ridge regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


weather = pd.read_csv("weather.csv", index_col="DATE")

null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
valid_columns = weather.columns[null_pct < 0.05]  # taking the valid columns with null values percentage less than 0.05
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()  # filling the valid columns with null with previous values
weather.index = pd.to_datetime(weather.index)  # converting the index to date format
weather.index.year.value_counts().sort_index()  # to get rid of gap entries

weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()

rr = Ridge(alpha=.1)  # alpha is lambda in ridge regression
# it controls how much the coefficients are shrunk to account for collinearity
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]


def backtest(weather, model, predictors, start=3650, step=90):  # starting with 10 years of data so
    # the prediction will be except first 10 years
    # and every 90 days we create a set of predictions
    # time series cross_validation is used for
    all_predictions = []  # each element will be a data frame of 90 days prediction
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]  # all before current row
        test = weather.iloc[i:(i + step):]  # all after i to 90 days

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])  # returns a numpy array

        preds = pd.Series(preds, index=test.index)  # converting to pandas series

        combined = pd.concat([test["target"], preds], axis=1)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions, axis=0)


#  to improve accuracy
def pct_diff(old, new):
    return (new - old) / old  # percentage change


def compute_rolling(weather, horizon, col):  # find the rolling average
    # horizon-->no of days we want to calculate the rolling avg for
    # col -->column name we want to get the rolling avg for
    label = f"rolling_{horizon}_{col}"  # name of the new column in dataframe
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])  # percentage diff between current day and rolling
    return weather


rolling_horizons = [3, 15]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)
weather = weather.iloc[14:, :]  # removes first 14 values
weather = weather.fillna(0)  # find missing values with 0


def expand_mean(df):
    return df.expanding(1).mean()


for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    # group by month
    # take all temp from same month from previous years and average them
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictions = backtest(weather, rr, predictors)
print(predictions)
print(mean_absolute_error(predictions["actual"], predictions["prediction"]))
predictions.sort_values("diff", ascending=False)
predictions["diff"].round().value_counts().sort_index().plot()
