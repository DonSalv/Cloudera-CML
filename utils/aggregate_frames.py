import pandas as pd
import numpy as np
from typing import Set

from os import listdir
from os.path import isfile, join

from utils import get_sla_right

PATH = '././data'


def get_time_series_variables_from(data: pd.DataFrame,
                                   filled: bool = True) -> pd.DataFrame:
    """
    Method to get the relevant variables for analysis of the Call Center data. It adds a calls_volume column for
    counting the registered calls.
    :param data: A monthly dataframe as in the bank data. Check for schema consistency.
    :param filled: A boolean parameter to indicate if missing values in SLA should be imputed with -1
    :return: A pandas dataframe with time calls measures
    """
    df = data.copy()

    # Useful variables for time-series analysis
    variables = ['start_date', 'abandon', 'prequeue', 'inqueue', 'agent_time', 'postqueue', 'total_time', 'sla',
                 'abandon_time']
    df = df[variables]

    # Add variable for the calls volume
    df['calls_volume'] = 1

    if filled:
        df['sla'].fillna(-1, inplace=True)

    return df


def add_time_features_from(column: str, data: pd.DataFrame,
                           date: bool = True, time: bool = True) -> pd.DataFrame:
    """
    A method to add time feature engineering variables from the date values of the dataframe

    :param column: The name of the column holding the date information. As timestamp variable type
    :param data: A monthly dataframe as the output of the get_time_series_variables_from method
    :param date: A boolean indicator if year, month and day must be added to the dataframe. Default = True
    :param time: A boolean indicator if hour and minute must be added to the dataframe. Default = True
    :return: A dataframe with time and date features
    """
    df = data.copy()

    if date:
        df['year'] = df[column].apply(lambda x: x.year)
        df['month'] = df[column].apply(lambda x: x.month)
        df['day'] = df[column].apply(lambda x: x.day)

    if time:
        df['hour'] = df[column].apply(lambda x: x.hour)
        df['minute'] = df[column].apply(lambda x: x.minute)

    return df


def _get_missing_days_in(data: pd.DataFrame) -> Set:
    """
    A method to get the missing days in a dataframe to complete the range of days for the given month
    :param data: A monthly dataframe as the output of the add_time_features_from method
    :return: A set with given missing days in the dataframe for the corresponding month
    """
    month = data['month'].unique()[0]
    leap = pd.Timestamp(data['start_date'][0]).is_leap_year

    if month in [1, 3, 5, 7, 8, 10, 12]:
        days = 32
    elif month in [4, 6, 9, 11]:
        days = 31
    else:
        if leap:
            days = 30
        else:
            days = 29

    mt_days = set(range(1, days))
    df_days = set(data['day'].unique())

    missing_days = mt_days.difference(df_days)

    return missing_days


def complete_days_range_in_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    A method to input a row with zero values in the corresponding missing days of the dataframe.
    :param data: A monthly dataframe as the output of the add_time_features_from method
    :return: A monthly dataframe as the output of the add_time_features_from method with complete days
             range in the month
    """
    df = data.copy()
    missing_days = _get_missing_days_in(data=df)

    for day in missing_days:

        year = df['year'].unique()[0]
        month = df['month'].unique()[0]

        if month < 10:
            d_month = f"0{month}"
        else:
            d_month = month

        if day < 10:
            d_day = f"0{day}"
        else:
            d_day = day

        date = f"{year}-{d_month}-{d_day} 00:00:00"

        row = [date, 0, 0, 0, 0, 0, 0, 0, 0, 0, year, month, day, 0, 0]
        df.loc[-1] = row
        df.index = df.index + 1
        df = df.sort_index()

    return df


def get_day_hour_aggregation_variables_from(data: pd.DataFrame,
                                            month: bool = False) -> pd.DataFrame:
    """
    A method to aggregate a dataframe based on the day and the hour. It aggregates the columns with meaningful function
    according to the variable.
    :param data: A monthly dataframe as the output of the add_time_features_from method
    :param month: A boolean value if the month feature should be used in the aggregation. Default = False
    :return: An aggregated by day and hour (and month if set to true) dataframe with meaningful aggregation values
    """
    df = data.copy()
    aggregation = ['day', 'hour']

    if month:
        aggregation.append('month')

    df = df.groupby(aggregation).aggregate({"abandon": np.sum,
                                            "prequeue": np.mean,
                                            "inqueue": np.mean,
                                            "agent_time": np.mean,
                                            "postqueue": np.mean,
                                            "total_time": np.mean,
                                            "sla": pd.Series.mode,
                                            "calls_volume": np.sum})

    return df


def get_aggregated_daily_frame_for(day: int,
                                   data: pd.DataFrame) -> pd.DataFrame:
    """
    A method to quickly select an aggregated dataframe for a given day displaying hourly values for the variables
    :param day: The day of the given month of the dataframe. The day should be present in the dataframe
    :param data: A monthly dataframe as the output of the get_day_hour_aggregation_variables_from
    :return: A dataframe with hourly variable's information for the given day
    """
    df = data.copy()

    df = df.iloc[df.index.get_level_values('day') == day]

    return df


def complete_hour_range_in_frame(data: pd.DataFrame,
                                 day: int) -> pd.DataFrame:
    """
    A method to complete the 24 hours data schedule for the given day.
    :param data: A pandas dataframe as the output of the get_day_hour_aggregation_variables_from method
    :param day: The day of the month to complete the time schedule for 24 hours.
    :return: A complete dataframe with zero imputation for missing hours
    """
    df = get_aggregated_daily_frame_for(day=day, data=data)

    if 'month' in df.index.names:
        df_hours = set(df.loc[day, :, df.index[0][-1]].index.get_level_values(1))
    else:
        df_hours = set(df.loc[day, :].index.get_level_values(0))

    dy_hours = set(range(24))
    missing_hours = dy_hours.difference(df_hours)

    hour_values = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)

    for hour in missing_hours:
        if 'month' in df.index.names:
            df.loc[(day, hour, df.index[0][-1]), :] = hour_values
        else:
            df.loc[(day, hour), :] = hour_values

    test_cols = ['abandon', 'prequeue', 'inqueue', 'agent_time', 'postqueue', 'total_time']

    df['calls_volume'] = np.where(df[test_cols].any(axis=1), df['calls_volume'], 0)

    return df


def get_complete_day_hour_aggregation_variables_from(data: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    A method to complete the 24 hours data schedule for all the days in the month.
    :param year: The year of the current frame
    :param data: A pandas dataframe as the output of the get_day_hour_aggregation_variables_from method with month
                 parameter set to "True"
    :return: A complete dataframe with zero imputation for missing hours
    """
    df = pd.DataFrame(
        columns=['abandon', 'prequeue', 'inqueue', 'agent_time', 'postqueue', 'total_time', 'sla', 'calls_volume'])

    month = data.index[0][-1]
    leap = pd.Timestamp(f"{year}-01-01").is_leap_year

    if month in [1, 3, 5, 7, 8, 10, 12]:
        days = 32
    elif month in [4, 6, 9, 11]:
        days = 31
    else:
        if leap:
            days = 30
        else:
            days = 29

    for day in range(1, days):
        df = pd.concat([df, complete_hour_range_in_frame(data=data, day=day).reset_index()])

    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)

    if 'month' in data.index.names:
        df['month'] = df['month'].astype(int)
        df.set_index(['day', 'hour', 'month'], inplace=True)
    else:
        df.set_index(['day', 'hour'], inplace=True)

    return df


def get_aggregated_hourly_frame_for(hour: int,
                                    data: pd.DataFrame) -> pd.DataFrame:
    """
    A method to quickly select an aggregated dataframe for a given day displaying hourly values for the variables
    :param hour: The hour of the day for the given month of the dataframe. The hour should be present in the dataframe
    :param data: A monthly dataframe as the output of the get_day_hour_aggregation_variables_from
    :return: A dataframe with monthly data variable's information for the given hour
    """
    df = data.copy()

    df = df.iloc[df.index.get_level_values('hour') == hour]

    return df


def get_yearly_frame() -> pd.DataFrame:
    monthly_records = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    df = pd.DataFrame(
        columns=['abandon', 'prequeue', 'inqueue', 'agent_time', 'postqueue', 'total_time', 'sla', 'calls_volume'])

    for record in monthly_records:
        data = pd.read_csv(f"{PATH}/{record}", parse_dates=['start_date'])
        aux = get_time_series_variables_from(data=data)
        aux = add_time_features_from(column='start_date', data=aux)
        year = aux['year'][0]
        aux = complete_days_range_in_frame(data=aux)
        aux = get_day_hour_aggregation_variables_from(data=aux, month=True)
        aux = get_sla_right(aux)
        aux = get_complete_day_hour_aggregation_variables_from(data=aux, year=year)

        df = pd.concat([df, aux.reset_index()])

    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)
    df['month'] = df['month'].astype(int)
    df.set_index(['day', 'hour', 'month'], inplace=True)

    return df


if __name__ == '__main__':
    # dataframe = pd.read_csv("Data/01-january.csv",
    #                         parse_dates=['start_date'])
    #
    # df_aux = get_time_series_variables_from(data=dataframe)
    # df_aux = add_time_features_from(column='start_date',
    #                                 data=df_aux)
    # df_aux = get_day_hour_aggregation_variables_from(data=df_aux)
    #
    # day_of_month = int(input("Input the day of the month: "))
    # df1st = get_aggregated_daily_frame_for(day=day_of_month,
    #                                        data=df_aux)
    # print(df1st.sample(8, random_state=42))

    print(get_yearly_frame())
