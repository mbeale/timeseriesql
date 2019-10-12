import os
import time
from .timeseries import TimeSeries
from .decompiler import Decompiler


class Plan:
    """A basic class that holds the Plan execution details"""

    __slots__ = ["calc", "group", "metrics", "filters", "variables"]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
        else:
            start_offset = (
                self.period.start
                if self.period and self.period.start
                else self.DEFAULT_START_OFFSET
            )
            end_offset = self.period.stop if self.period and self.period.stop else 0
            resolution = (
                self.period.step if self.period and self.period.step else self.DEFAULT_RESOLUTION
            )

        period = {
            "start_time": now - start_offset,
            "end_time": now - end_offset,
            "resolution": resolution,
        }
        return period

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
        plans = []
        for g in self.generators:
            plan = Plan(
                **{
                    "calc": None,
                    "group": self.groupings,
                    "metrics": [],
                    "filters": self.filters,
                    "variables": [],
                }
            )
            plans.append(Decompiler(g, plan).decompile())

        return plans

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
        return self.__getitem__(None)
