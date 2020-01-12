import os
import time
import numbers
from .timeseries import TimeSeries
from .time import convert_string_to_seconds
from .ast import AST


class Query:
    """
    A query class that generates an execution plan. This class can be 
    easily extended by overriding the `execute_plan` plan method.
    """

    DEFAULT_START_OFFSET = int(os.environ.get("DEFAULT_START_OFFSET", "3600"))
    DEFAULT_RESOLUTION = int(os.environ.get("DEFAULT_RESOLUTION", "60"))

    def __init__(self, *args):
        """Override __init__"""
        self.groupings = None
        self.period = None
        self.filters = []
        self.data = None
        self.generators = [a for a in args]

    def __iter__(self):
        """Make Query object an iterable"""
        return self

    def __next__(self):
        """Nothing to iterate"""
        raise StopIteration()

    def __getitem__(self, items):
        """Have the ability to fetch data via the getitem method

        Example
        -------

        t1 = Query(x.mean for x in "test")[x:y:z]

        params
        ------

        items: slice
            start = start_time_offset in seconds (now - start_time_offset = start_time)
            stop = end_time_offset in seconds (now - end_time_offset = end_time)
            step = resolution in seconds
        """
        self.period = items
        return self.execute_plan()

    def execute_plan(self):
        """Execute the query against the backend

        In this case there is a simple return of an empty
        TimeSeries with the shape of the calculated time period.
        This should be overridden by a backend to process the query.

        """
        period = self._process_period()
        # should return time series
        # return empty timeseries based on the period
        timeindex = [
            x for x in range(period["start_time"], period["end_time"], period["resolution"])
        ]
        t1 = TimeSeries(shape=(len(timeindex), 1), time=timeindex)
        # t1[:] = [[i,np.nan] for i in timeindex]
        return t1

    def _process_period(self):
        """ This processes the period from the __getitem__() function

        Slice Examples:
        --------------

        [3600] start_time = now - 3600 seconds, end_time=now
        [3600:1800] start_time = now - 3600 seconds, end_time = now - 1800 seconds
        [3600:1800:300] start_time = now - 3600 seconds, end_time = now - 1800 seconds, resolution = 5 minutes

        String Examples
        ---------------
        ['15m'] start_time = now - 900 seconds

        date string options are:

        s - seconds
        m - minutes
        h - hours
        d - days
        w - weeks
        y - years

        Returns
        -------
        a period dictionary
        """
        start_offset = self.DEFAULT_START_OFFSET
        end_offset = 0
        resolution = self.DEFAULT_RESOLUTION
        now = int(time.time())
        if isinstance(self.period, int):
            start_offset = self.period
        elif isinstance(self.period, slice):
            start_offset = (
                self.period.start
                if self.period and self.period.start
                else self.DEFAULT_START_OFFSET
            )
            end_offset = self.period.stop if self.period and self.period.stop else 0
            resolution = (
                self.period.step if self.period and self.period.step else self.DEFAULT_RESOLUTION
            )
        elif isinstance(self.period, str):
            try:
                start_offset = convert_string_to_seconds(self.period)
            except:
                self._raise_period_index_error()
        else:
            self._raise_period_index_error()

        period = {
            "start_time": now - start_offset,
            "end_time": now - end_offset,
            "resolution": resolution,
        }
        return period

    def _raise_period_index_error(self):
        raise IndexError(
            f"Only slices, integers, and date format strings are allowed not {type(self.period)}"
        )

    def _generate_plan(self):
        """This function generates a plan for the query engine to execute

        Return Example
        --------------

        {
                "metrics": [
                    {
                        "eval": None,
                        "metrics": ["metric1"],
                        "filters": [],
                        "group": [['tag1'], 'mean'],
                        "vars": [{'name': 'x', 'labels': ['max']}]
                    },{
                        "eval": None,
                        "metrics": ["metric2"],
                        "filters": [],
                        "group": [['tag1'], 'mean'],
                        "vars": [{'name': 'x', 'labels': ['min']}]
                    }
                ],
                "eval": [['x', 'y', 'BINARY_SUBTRACT']],
                "filters": [],
                "group": None,
                "vars": [{'name': 'x', 'labels': []}, {'name': 'y', 'labels': []}]
        }
        """
        return AST(self.generators, self.groupings).decompile()

    def by(self, labels, func="mean"):
        """Adds a group by step to the Query plan
        
        Params:
        -------
        Labels: list of strings
            The labels that are to be grouped on
        func:  a ufunc
            The function to merge the different stream with

        Returns
        -------
        Self
        """
        if not isinstance(labels, list):
            labels = [labels]
        if not self.groupings:
            self.groupings = []
        self.groupings = [labels, func]
        return self

    def fetch(self):
        """Fetch all items.  This can be overridden with sane defaults"""
        return self[:]

    def range(self, start, end=None, resolution=None):
        """ 
        Fetch points by a specified time range 
        
        Params:
        -------
        start: int
            A unixtimestamp for the start
        end: int
            A unixtimestamp for the end
        resolution: int
            A int for the resolution

        Returns
        -------
        TimeSeries
        """
        now = int(time.time())
        end_offset = None if end == None else end - start
        res = self.DEFAULT_RESOLUTION if resolution is None else resolution

        s = slice(now - start, end_offset, res)
        return self[s]
