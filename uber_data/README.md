# Uber Pickup Data For New York City

We explore and analyze Uber pick up data for New York City.
[The data is from FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response).
The data includes the following for each pickup:

* Date/Time
* Latitude
* Longitude

# Virtual Environment

Notebooks use `statsmodels.py` which has a dependency on `scipy.py 1.2.0`. So we need
to use a virtual environment. 

## Installing the Virtual Environment

Make sure you have `virtualenv.py`; if not, then run
```
pip install virtualenv
```

First, we install the proper version of `scipy` in the virtual environment and install a `jupyter`
kernel for the virtual environment. From the project directory, run the following:

1. Run `virtualenv --system-site-packages venv`.
2. Activate the virtual environment. For example on Windows, run `venv\Scripts\activate`.
3. From the virtual environment, run `pip install scipy==1.2.0`. 
4. From the virtual environment, run `python -m ipykernel install --user --name=uber_data`.
5. Exit the virtual environment by running `deactivate`.

## Using the Virtual Environment

Now, when we run `jupyter notebook` (inside or outside the virtual environment), just make sure
you are using the `uber_data` kernel (and NOT the default kernel, e.g. `Python 3`).

For example, using the menus in the notebook, go to `Kernel > Change kernel > uber_data`.

# Notebooks

We have the following notebooks:

## Hourly Pickup Counts for Different Days of the Week

###[`continuous_weekly_negative_binomial.ipynb`](continuous_weekly_negative_binomial.ipynb)

Group the pickup data to get counts of how many pickups occur each hour (so 24 in a single day).
Model these counts based on the day of the week. Can be used to make strategic decisions related to
how busy drivers are at different points in the week (e.g. what are the most popular times of the week).

Here is the resulting model:

![Final Model Counts](graphs/final_model_counts.svg?sanitize=true)

The model was selected from several variations using cross-validation where the hold-out set is
scored based on the mean log-likelihood of the model.

## Hourly Pickup Exploration

### [`hourly_pickups_exploration.ipynb`](hourly_pickups_exploration.ipynb)

Group the pickup data by the date and hour. Look at trends in the counts of the number of
pickups for a given hour in the day. Look at splitting by days that are weekends, weekdays, etc.
For example, we have the following for hourly counts for Monday through Thursday (Friday nights
should have a different behavior).

![Hourly Pickups for Monday Through Thursday](graphs/mon_th_pickups.svg?sanitize=true)
