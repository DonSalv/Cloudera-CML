import pandas as pd
from darts import TimeSeries


from darts.metrics import coefficient_of_variation
from matplotlib import pyplot as plt
from typing import Tuple


def get_sla_right(data: pd.DataFrame) -> pd.DataFrame:
    """
    An auxiliary method the remove the array entries in the SLA variable. It is replaced by -1.0 being strict with the
    value for those hours not meeting the SLA
    :param data: A pandas dataframe as the get_yearly_frame()
    :return: A pandas dataframe with strict considerations in the SLA column.
    """
    df = data.copy()

    mask = df['sla'].apply(lambda x: not isinstance(x, float))

    df.loc[mask, 'sla'] = -1.0

    return df


def get_formatted_time_series_frame_from(data: pd.DataFrame, year: str) -> pd.DataFrame:
    """
    A method to get a complete, yearly dataframe by hours and ready to create Time Series objects in darts for
    analysis
    :param data: A pandas dataframe as the output of get_yearly_frame().
    :param year: The year to consider the yearly dataframe
    :return: A pandas dataframe with no missing values, right SLA formatting and date column
    """
    df = data.copy()
    df['year'] = year
    df.reset_index(inplace=True)

    df["date"] = df['year'] + '-' + df["month"].astype(str) + '-' + df["day"].astype(str) + ' ' + df['hour'].astype(
        str) + ':00:00'
    df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)

    df = get_sla_right(data=df)

    return df


def plot_predict(train_target: TimeSeries,
                 test_target: TimeSeries,
                 prediction: TimeSeries,
                 low_percentile: float = 0.05,
                 figure_size: Tuple[int, int] = (18, 6)):
    """
    Given a train, test and predicted values it returns the corresponding plot. If low_percentile is given, the plot
    includes the band percentiles around the median value of the stochastic forecast
    :param train_target: A time series with the train values
    :param test_target: A time series with the test values
    :param prediction: A time series with the predicted values
    :param low_percentile: A float number indicating the low percentile for the values. Default 0.05
    :param figure_size: The size of the plt figure. Default (18, 6)
    :return:
    """
    high_percentile = 1 - low_percentile
    title = f"Percentile band: ({low_percentile * 100:.0f}% - {high_percentile * 100:.0f})%"

    if prediction.is_deterministic:
        title = ""

    # Plot time series, limited to forecast horizon
    plt.figure(figsize=figure_size)

    train_target.plot(label="Train")

    test_target.plot(label="Test")

    prediction.plot(central_quantile=0.5,
                    low_quantile=low_percentile,
                    high_quantile=high_percentile,
                    label='Expected')

    plt.title(f"Actual values for train, test, and predicted values. {title}")

    plt.annotate(text=f"R-RMSE: {coefficient_of_variation(test_target, prediction):.2f}%",
                 xy=(0, 0),
                 xytext=(0, 10),
                 xycoords='figure pixels')

    return plt


def plot_prediction_and_test(target: TimeSeries,
                             prediction: TimeSeries,
                             figure_size: Tuple[int, int] = (18, 6)):
    """
    Given a target and predicted values it returns the corresponding plot.
    :param target: A time series with the train values
    :param prediction: A time series with the predicted values
    :param figure_size: The size of the plt figure. Default (18, 6)
    :return: A plot of the given Time Series.
    """
    plt.figure(figsize=figure_size)
    target.plot(label='Actual')
    prediction.plot(label='Expected')

    plt.title("Test vs Predicted - Hourly Calls Volume")
    plt.annotate(text=f"R-RMSE: {coefficient_of_variation(target, prediction):.2f}%",
                 xy=(0, 0),
                 xytext=(0, 5),
                 xycoords='figure pixels')

    return plt
