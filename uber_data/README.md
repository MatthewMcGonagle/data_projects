# Uber Pickup Data For New York City

We explore and analyze Uber pick up data for New York City.
[The data is from FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response).
The data includes the following for each pickup:

* Date/Time
* Latitude
* Longitude

# Notebooks

We have the following notebooks:

## Hourly Pickup Exploration

### [`hourly_pickups_exploration.ipynb`](hourly_pickups_exploration.ipynb)

Group the pickup data by the date and hour. Look at trends in the counts of the number of
pickups for a given hour in the day. Look at splitting by days that are weekends, weekdays, etc.
For example, we have the following for hourly counts for Monday through Thursday (Friday nights
should have a different behavior).

![Hourly Pickups for Monday Through Thursday](graphs/mon_th_pickups.svg?sanitize=true)
