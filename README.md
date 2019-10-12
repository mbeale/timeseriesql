
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
  * [CSV Backend](#csv_backend_usage)
  * [AppOptics Backend](#ao_backend_usage)
  * [TimeSeries](#timeseries_usage)
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
* [Pandas](https://pandas.pydata.org)
* [Matplotlib](https://matplotlib.org)



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
data = Query(x for x in "metric.name" if x.some_label = "some_value").group("a_label")[start:end:resolution]
```

The return value is a ``TimeSeries`` object that uses a Numpy array as backend.  That object can have 
``ufuncs`` and other numpy functions applied against it.  More examples to come.

### CSV Backend Usage

Often time data is loaded from a CSV file.  The backend expects the first column to be the time index in 
either a numerical timestamp or strings in ISO 8601 date or datetime format.  The filters are applied to the
headers of the CSV.  If labels are not in the CSV and are supplied as part of the query, then filters will not
be applied.

If any columns are empty or don't contain a numeric value, the value becomes a ``np.nan``.


#### Basic CSV Usage

```python
data = CSVBackend(x for x in "path/to.csv")[:]
```

#### Basic CSV Filtering

```python
data = CSVBackend(x for x in "path/to.csv" if x.label == "A")[:]
data = CSVBackend(x for x in "path/to.csv" if x.label != "B")[:]
data = CSVBackend(x for x in "path/to.csv" if x.label in ["B", "C", "G"])[:]
data = CSVBackend(x for x in "path/to.csv" if x.label not in ["B", "C", "G"])[:]
```

### Set the Labels

```python
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
