import numpy as np
import csv
from .query import Query
from .timeseries import TimeSeries


class CSVBackend(Query):

    # handle url or local file
    # (x for x in "csvfilename" x.fieldname = fieldname)
    # handle headers

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    def execute_plan(self):
        params = self._process_period()
        plan = self._generate_plan()[0]
        labels = []
        for m in plan.metrics:
            with open(m) as csvfile:
                filereader = csv.reader(csvfile)
                read_header = False
                for row in filereader:
                    if not read_header:
                        labels = row[1:]
                        read_header = True
                        continue

        t = TimeSeries(shape=(1, 1), time=[0], labels=labels)
        return t
