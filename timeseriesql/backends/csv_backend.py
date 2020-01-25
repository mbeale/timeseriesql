import numpy as np
import csv
import datetime
from dateutil.parser import parse
from itertools import compress
from timeseriesql.query import Query
from timeseriesql.timeseries import TimeSeries
from timeseriesql.ast import Metric, Value


def convert_to_float(s):
    try:
        return float(s)
    except:
        return np.nan


def convert_date_to_float(s):
    try:
        return float(s)
    except:
        d = parse(s)
        return datetime.datetime.timestamp(d)


class CSVFile:
    def __init__(self, name):
        self.name = name
        self.filters = []


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
                        if x != f["right"]:
                            mask[i] = 0
                    elif f["op"] == "!=":
                        if x == f["right"]:
                            mask[i] = 0
                    elif f["op"] == "in":
                        if x not in f["right"]:
                            mask[i] = 0
                    elif f["op"] == "not in":
                        if x in f["right"]:
                            mask[i] = 0
                    else:
                        raise NotImplementedError(
                            f"unsupported op type of '{f['op']}' for the filter"
                        )
        return mask

    def compareequal(self, left, right):
        return {"op": "==", "right": right}

    def comparenotequal(self, left, right):
        return {"op": "!=", "right": right}

    def comparein(self, left, right):
        return {"op": "in", "right": right}

    def comparenotin(self, left, right):
        return {"op": "not in", "right": right}

    def filter(self, left, right):
        left.filters.append(right)
        return left

    def traverse_tree(self, root):
        if root:
            if isinstance(root, Metric):
                return CSVFile(root.value)
            if isinstance(root, Value):
                return root.value
            else:
                left = self.traverse_tree(root.left)
                right = self.traverse_tree(root.right)
                try:
                    op = self.__getattribute__(root.__class__.__name__.lower())
                    return op(left, right)
                except Exception as e:
                    raise NotImplementedError(
                        f"AST class of {root.__class__.__name__} is not supported"
                    )

    def execute_plan(self):
        ast = self._generate_plan()
        csvobject = self.traverse_tree(ast)
        labels = []
        data = []
        time = []
        if str(type(csvobject.name)) == "<class '_csv.reader'>":
            csvfile = None
            filereader = csvobject.name
        else:
            csvfile = open(csvobject.name)
            filereader = csv.reader(csvfile)
        read_header = False
        mask = None
        for row in filereader:
            if not read_header:
                if not self._noheader:
                    read_header = True
                    if not self._labels:
                        mask = self._header_mask(csvobject.filters, row)
                        labels = [{"label": x} for x in compress(row, mask)]
                    else:
                        mask = [0] + [1 for x in row[1:]]
                        labels = self._labels
                    continue
                else:
                    mask = [0] + [1 for x in row[1:]]
                    labels = self._labels
                    read_header = True
            data.append(list(map(convert_to_float, compress(row, mask))))  # row zero is time
            time.append(row[0])
        t = TimeSeries(
            shape=(len(data), len(labels)),
            time=list(map(convert_date_to_float, time)),
            labels=labels,
        )
        if csvfile:
            del csvfile
        t[:] = data
        return t[self.period]

