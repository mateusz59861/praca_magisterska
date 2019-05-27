import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime, time


def count_outliers(dataa):
    return len(np.where((dataa > 0.5) | (dataa < -0.5))[0])

	
def calculate_trend(data, timestamps):
    dates_numbers = []
    for date in timestamps:
        dates_numbers.append(time.mktime(date.timetuple()))

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(np.array(dates_numbers).reshape(-1, 1), data)  # perform linear regression
    calculated_trend = linear_regressor.predict(np.array(dates_numbers).reshape(-1, 1)) # make predictions
    return calculated_trend