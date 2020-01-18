
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">TimeSeriesQL</h3>

  <p align="center">
    A Pythonic query language for time series data
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [CSV Backend](#csv-backend-usage)
  * [TimeSeries](#timeseries-usage)
* [Plotting Libs](#plotting_libs)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

There are many time series databases and each have their own query language.  Each platform takes time to invest in 
learning the structure and keywords of that language and often the skills learned don't translate to other 
platforms.  The goal of this project is to create a time series specific library that can be used across many different time series databases as well as easy to learn because it uses Python syntax.  


### Built With

* [Numpy](https://numpy.org)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

The requirements are in the [requirements.txt](requirements.txt) file.

### Installation

#### pip

```sh
pip install timeseriesql
```

#### manual

1. Clone the timeseriesql
```sh
git clone https:://github.com/mbeale/timeseriesql.git
```
2. Install library
```sh
cd timeseriesql
python setup.py install 
```

<!-- USAGE EXAMPLES -->
## Usage

The way this project works is to provide a general framework for querying a time series with pluggable
backends that communicate with specific time series databases.  The queries are created using Python
generators, a formatt familiar to Pythonistas.

```python
data = Query(x for x in "metric.name" if x.some_label = "some_value").by("a_label")[start:end:resolution]
```

The return value is a ``TimeSeries`` object that uses a Numpy array as backend.  That object can have 
``ufuncs`` and other numpy functions applied against it.  More examples to come.

There are defaults for start and resolution that are controlled by environment variables.  That helps avoid 
fetching all measurements from the beginning of time by accident.

```sh
DEFAULT_START_OFFSET #defaults to 3600 seconds
DEFAULT_RESOLUTION   #defaults to 60 seconds
```

### CSV Backend Usage

Often time series data is loaded from a CSV file.  The backend expects the first column to be the time index in 
either a numerical timestamp or strings in ISO 8601 date or datetime format.  The filters are applied to the
headers of the CSV.  If labels are not in the CSV and are supplied as part of the query, then filters will not
be applied.

If any columns are empty or don't contain a numeric value, the value becomes a ``np.nan``.


#### Basic CSV Usage

```python
from timeseriesql.backends.csv_backend import CSVBackend

data = CSVBackend(x for x in "path/to.csv")[:]
```

#### Basic CSV Filtering

For CSV files the labels are the column headers.  If there are columns that are not needed, they can be filtered out.

```python
from timeseriesql.backends.csv_backend import CSVBackend

data = CSVBackend(x for x in "path/to.csv" if x.label == "A")[:]
data = CSVBackend(x for x in "path/to.csv" if x.label != "B")[:]
data = CSVBackend(x for x in "path/to.csv" if x.label in ["B", "C", "G"])[:]
data = CSVBackend(x for x in "path/to.csv" if x.label not in ["B", "C", "G"])[:]
```

#### Set the Labels

```python
from timeseriesql.backends.csv_backend import CSVBackend

data = CSVBackend(x for x in "path/to.csv").labels(
    [
        {"label": "one"},
        {"label": "two"},
        {"label": "three"},
        {"label": "four"},
        {"label": "five"},
        {"label": "six"},
        {"label": "seven"},
    ]
)[:]
```

### TimeSeries Usage

The `TimeSeries` object is allows for manipulation of the time series data after the it's been queried from the 
backend.   

In the following examples, the variables starting with `ts_` are assumed to be queried data from a backend.

#### TimeSeries Operations

```python

# Basic mathematical operations (+, -, /, *)
ts_1 + 5 # will return a new series
ts_1 += 5 #will perform operation in place
ts_1 += ts_2 #add together two TimeSeries

```

#### TimeSeries Time Index

The time index is a array of floats but there is a built in method to convert the floats into `np.datetime64`.

```python
ts_1.time # array of floats
ts_1.time.dt #array of np.datetime64
```

#### TimeSeries Merging

`TimeSeries` objects can be combined but the ending time indexes must be the same.  This may require empty values 
to be created where the indexes don't align.

```python
new_t = ts_1.merge([ts_2, ts_3])
```

#### TimeSeries Grouping/Reducing

If there are multiple streams, they can be grouped and merged by the labels.

```python
reduced = ts_1.group(["hostname", "name"]).add() 
reduced = ts_1.group("env").mean()
reduced = ts_1.group("env").mean(axis=None) #setting the access to None will get the mean of the entire object
```

#### TimeSeries Special Indexing

```python
import numpy as np

beg = np.datetime64('2019-02-25T03:00')
end = np.datetime64('2019-02-25T04:00')

ts_1[beg:end] # set a time range
ts_1[beg : np.timedelta64(3, "m")] # fetch from beginning + 3 minutes
ts_1[np.timedelta64(3, "m") :] #start from beginning + 3 minutes
ts_1[: np.timedelta64(3, "m")] #end at the end - 3 minutes


ts_1[{"hostname": "host2"}] # by labels

```

#### TimeSeries Rolling Windows

The `rolling_window` method assumes that the data is filled and at a fixed resolution.  Number of periods is 
an integer and not a time range.

```python
rolling_cum_sum = ts_1.rolling_window(12).add() #rolling cumsum
rolling_mean = ts_1.rolling_window(12).mean() # rolling mean
rolling = ts_1.rolling_window(12).median() #rolling median
```

#### TimeSeries Resample

The `resample` method allows a smaller period to be aggregated into a larger period.

```python
resampled = ts_1.resample(300).mean() #resamples to 5 minutes and takes the mean
```

#### TimeSeries to Pandas

The conversion returns 2 pandas DataFrames, one for the labels and the other for the data.

```python
data, labels = ts_1.to_pandas()
```

<!-- PLOTTING_LIBS -->
## Plotting Libs

### Available

* [matplotlib](https://github.com/mbeale/tiemseriesql-matplotlib)

### Creating a custom backend

Start by extending the [Plot](https://github.com/mbeale/timeseriesql/blob/master/timeseriesql/plot.py) class.

```python
from timeseries.plot import Plot
class NewPlottingLib(Plot):
  pass
```

There is a list of functions that can be extended for as different plots.  Also there are functions that generate titles, xlabel, ylabel, and legend labels.  Use those to grab default information.  They can be overridden to provide more custom logic around the those fields.


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/mbeale/timeseriesql/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Michael Beale - michael.beale@gmail.com

Project Link: [https://github.com/mbeale/timeseriesql](https://github.com/mbeale/timeseriesql)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/mbeale/timeseriesql.svg?style=flat-square
[contributors-url]: https://github.com/mbeale/timeseriesql/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mbeale/timeseriesql.svg?style=flat-square
[forks-url]: https://github.com/mbeale/timeseriesql/network/members
[stars-shield]: https://img.shields.io/github/stars/mbeale/timeseriesql.svg?style=flat-square
[stars-url]: https://github.com/mbeale/timeseriesql/stargazers
[issues-shield]: https://img.shields.io/github/issues/mbeale/timeseriesql.svg?style=flat-square
[issues-url]: https://github.com/mbeale/timeseriesql/issues
[license-shield]: https://img.shields.io/github/license/mbeale/timeseriesql.svg?style=flat-square
[license-url]: https://github.com/mbeale/timeseriesql/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/michael-beale-163a4670
