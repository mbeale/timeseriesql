import numpy as np
import csv
from itertools import compress
from timeseriesql.query import Query
from timeseriesql.timeseries import TimeSeries


def convert_to_float(s):
    try:
        return float(s)
    except:
        return np.nan


def convert_date_to_float(s):
    try:
        return float(s)
    except:
        return np.datetime64(s).astype(float)


class CSVBackend(Query):

    _labels = None
    _noheader = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    def labels(self, labels, noheader=False):
        self._labels = labels
        self._noheader = noheader
        return self

    def _header_mask(self, filters, row):
        mask = [0] + [1 for x in row[1:]]
        if len(filters) > 0:
            for f in filters:
                if not any(mask):
                    break
                for i, x in enumerate(row[1:], 1):
                    if mask[i] == 0:
                        continue
                    if f["op"] == "==":
                        if x != f["right"].value:
                            mask[i] = 0
                    elif f["op"] == "!=":
                        if x == f["right"].value:
                            mask[i] = 0
                    elif f["op"] == "in":
                        if x not in f["right"].value:
                            mask[i] = 0
                    elif f["op"] == "not in":
                        if x in f["right"].value:
                            mask[i] = 0
                    else:
                        raise NotImplementedError(
                            f"unsupported op type of '{f['op']}' for the filter"
                        )
        return mask

    def execute_plan(self):
        plan = self._generate_plan()[0]
        labels = []
        data = []
        time = []
        for m in plan.metrics:
            with open(m) as csvfile:
                filereader = csv.reader(csvfile)
                read_header = False
                mask = None
                for row in filereader:
                    if not read_header:
                        if not self._noheader:
                            read_header = True
                            if not self._labels:
                                mask = self._header_mask(plan.filters, row)
                                labels = [{"label": x} for x in compress(row, mask)]
                            else:
                                mask = [0] + [1 for x in row[1:]]
                                labels = self._labels
                            continue
                        else:
                            mask = [0] + [1 for x in row[1:]]
                            labels = self._labels
                            read_header = True
                    data.append(
                        list(map(convert_to_float, compress(row, mask)))
                    )  # row zero is time
                    time.append(row[0])
        t = TimeSeries(
            shape=(len(data), len(labels)),
            time=list(map(convert_date_to_float, time)),
            labels=labels,
        )
        t[:] = data
        return t
