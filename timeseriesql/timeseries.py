import numpy as np
import numbers
from collections.abc import Sequence
from .np_array import NumpyArray
from .time import TimeIndex, convert_string_to_seconds
from itertools import compress
from .timeseries_collection import TimeSeriesCollection
from .time_chunk import TimeChunk
from .utils import dispatchmethod


class TimeSeries:
    """ The base object to work with time series 
    
    Attributes
    ----------
    data : NumpyArray
        The data associated with the time series object
    _time : np.array
        Then shared time index for the time series
    labels : [dict]
        A list of key value pairs for the streams
    """
    def __init__(self, *args, shape=(4, 3), labels=None, time=[], **kwargs):
        """
        Initialize the object

        Parameters
        ----------
        shape : tuple
            the shape to use to make the data array
        labels : [dict]
            a list of key value pairs for the streams
        time : array like
            a list of time indicies
        """
        self.data = NumpyArray(shape, self)
        if isinstance(labels, list):
            self.labels = [l for l in labels]
        elif labels:
            self.labels = [labels]
        else:
            self.labels = []
        if time != []:
            time = time[-self.data.shape[0]:]
        self._time = np.array(time, dtype=np.float64)

    def __array__(self):# pragma: no cover
        return self.data

    def __array_wrap__(self, out_arr, context=None):
        return self.wrap_new_data(out_arr)

    def __str__(self):# pragma: no cover
        """ String override which gives a larger amount of details """
        labels = ""
        for i, l in enumerate(self.labels):
            if i > 4:
                labels += f"{len(self.labels) - 5} more....\n"
                break
            for k, v in l.items():
                labels += f"{k}:{v} "
            labels += "\n"
        if labels == "":
            labels = None
        time_range = ""
        if self.time[0] > 0:
            time_range = f"Time Range\n===========\n{self.time.dt[0]} ... {self.time.dt[-1]}\n\n"
        return f"\nLabels:\n===========\n{labels}\n{time_range}Data:\n============\n{self.data.__str__()}"

    def __getattr__(self, attr_name):
        """ Attach the data attributes to the top level """
        if attr_name in dir(self.data) and not callable(getattr(self.data, attr_name)):
            return getattr(self.data, attr_name)

        def method(*args, **kwargs):
            if attr_name in dir(self.data):
                return getattr(self.data, attr_name)(*args, **kwargs)
            if attr_name in dir(np) and callable(getattr(np, attr_name)):
                return getattr(np, attr_name)(self, *args, **kwargs)
            else:
                raise AttributeError

        return method

    def __getitem__(self, items):
        """Add functionality for filtering on labels and datetime objects"""
        # if a label filter, return view for the correct streams
        try:
            time_index = self._convert_time_index(items)
            return self.wrap_new_data(self.data.__getitem__(items), time=self.time[time_index], ndx=items)
        except IndexError as e:
            index = self._handle_index_error(items, e)
            time_index = self._convert_time_index(index)
            return self.wrap_new_data(self.data.__getitem__(index), time=self.time[time_index], ndx=index)
        except TypeError as e:
            index = self._handle_type_error(items, e)
            time_index = self._convert_time_index(index)
            return self.wrap_new_data(self.data.__getitem__(index), time=self.time[time_index], ndx=index)

    def __setitem__(self, key, item):
        """ Set the item based off of a key in data """
        try:
            self.data.__setitem__(key, item)
        except IndexError as e:
            return self.data.__setitem__(self._handle_index_error(key, e), item)
        except TypeError as e:
            return self.data.__setitem__(self._handle_type_error(key, e), item)

    @dispatchmethod
    def _handle_index_error(self, key, error):
        l = len(key)
        if l != 1:
            raise error
        if l == 1:
            filters = self.argfilter(key)
            if len(filters) < 1:
                return slice(0,0)  # maybe should return something else?
            return (slice(None), filters)

    @_handle_index_error.register(int)
    def _(self, key, error):
        raise error

    @_handle_index_error.register(tuple)
    @_handle_index_error.register(slice)
    @_handle_index_error.register(np.datetime64)
    def _(self, key, error):
        time_slice = key
        extra_slices = None
        if isinstance(key, tuple):
            time_slice, extra_slices = key
        start, stop = self._get_slice_by_datetime(time_slice)
        if (start, stop) == (None, None):
            raise error
        if extra_slices:
            return (start, extra_slices)
        return start

    def _handle_type_error(self, key, error, item = None):
        # build new slice
        time_slice = key
        extra_slices = None
        if isinstance(key, tuple):
            time_slice, extra_slices = key
        if isinstance(time_slice, slice):
            start, stop = self._get_slice_by_datetime(time_slice.start, time_slice.stop)
            if (start, stop) == (None, None):
                raise error
            if extra_slices:
                return (slice(start, stop), extra_slices)
            else:
                return slice(start, stop)
        raise error

    def _convert_time_index(self, index):
        if not isinstance(index, Sequence):
            return index
        else:
            return index[0]

    def _merge_time_indicies(self, t1, t2):
        """Merge time indicies to crate an a new index of the union of the two indicies"""
        res = sorted(set(t1).union(set(t2)))
        return np.array(res)

    def _get_group_masks(self, group):
        """Return a mask for each label"""
        if not isinstance(group, list):
            group = [group]
        groupings = {}
        for i, stream in enumerate(self.labels.copy()):
            key = ""
            for g in group:
                if g in stream:
                    key += stream[g]
            if key in groupings:
                groupings[key].append(i)
            else:
                groupings[key] = [i]
        return [x for x in groupings.values()]

    def _get_slice_by_datetime(self, start_date=None, stop_date=None):
        """Return the indicies for a TimeSeries object searching by a datetime object"""
        start, stop = (start_date, stop_date)
        if isinstance(start_date, np.timedelta64):
            start_date = self.time.dt[0] + start_date
        if isinstance(start_date, np.datetime64):
            start = np.argwhere(self.time >= start_date.astype(float))
            if len(start) > 0:
                start = start[0][0]
            else:
                raise IndexError("No timestamps exist at or after %s" % start_date)
        if isinstance(stop_date, np.timedelta64):
            if start_date:
                stop_date = start_date + stop_date
            else:
                stop_date = self.time.dt[-1] - stop_date
        if isinstance(stop_date, np.datetime64):
            stop = np.argwhere(self.time >= stop_date.astype(float))
            if len(stop) > 0:
                stop = stop[0][0]
            else:
                stop = np.argwhere(self.time <= stop_date.astype(float))
                if len(stop) > 0:
                    stop = None
                else:
                    raise IndexError("No timestamps exist before or at %s" % stop_date)
        return (start, stop)

    def _least_common_labels(self, mask):
        """ create true false mask - probably a better way to do this """
        label_mask = []
        next_i = 0
        if isinstance(mask, slice):
            mask = [x for x in range(0,self.data.shape[1])]
        for m in mask:
            if next_i == m:
                label_mask.append(1)
                next_i += 1
            else:
                for _ in range(next_i,m):
                    label_mask.append(0)
                label_mask.append(1)
                next_i = m + 1
        labels = list(compress(self.labels.copy(), label_mask))
        common_labels = []
        if labels:
            common_labels = labels[0].copy()
            if len(labels) > 1:
                for l in labels[1:].copy():
                    for k in common_labels.copy().keys():
                        if k not in l or l[k] != common_labels[k]:
                            del common_labels[k]
        return common_labels

    def _get_unique_keys(self):
        """Get list of unique keys"""
        labels = self.labels[0].copy()
        for l in self.labels[1:]:
            for k in labels.copy().keys():
                if k not in l or l[k] != labels[k]:
                    del labels[k]
        return labels
    
    @property
    def time(self):
        """Return only the TimeSeries index"""
        return TimeIndex(self._time.view(np.ndarray))

    def wrap_new_data(self, out_arr, time=None, ndx=None):
        """ Wrap a value returned from a ufunc in a TimeSeries object """
        if not isinstance(out_arr, np.ndarray) or not out_arr.shape:
            return out_arr
        if time is None:
            time = self.time
        if not isinstance(time, np.ndarray):
            time = [time]
        if self.data.shape == out_arr.shape:
            new_to = TimeSeries(shape=out_arr.shape, time=time, labels=self.labels)
            new_to.data[:] = out_arr
        elif self.data.shape[0] == out_arr.shape[0]:
            rows = self.data.shape[0]
            columns = out_arr.shape[1] if len(out_arr.shape) > 1 else 1
            out_arr = out_arr.reshape((rows, columns))
            new_to = TimeSeries(shape=(rows, columns), time=time, labels=self.labels)
            new_to.data[:] = out_arr
            if ndx and len(ndx) > 1 and isinstance(ndx[1], (slice, int, list)):
                if isinstance(ndx[1], list):
                    labels = [self.labels[x] for x in ndx[1]]
                else:
                    labels = self.labels[ndx[1]]
                if not isinstance(labels, list):
                    labels = [labels]
                new_to.labels = labels
            else:
                label = self.parent._least_common_labels([x for x in range(0,self.parent.data.shape[1])])
                new_to.labels = [label for _ in range(columns)]
        elif self.data.shape[1] == out_arr.shape[0]:
            columns = self.data.shape[1]
            rows = out_arr.shape[0] if len(out_arr.shape) > 1 else 1
            if out_arr.size % (rows * columns) != 0: #not broadcastable
                new_to = TimeSeries(shape=out_arr.shape, time=time, labels=self.labels)
            else:
                out_arr = out_arr.reshape((rows, columns))
                new_to = TimeSeries(shape=(rows, columns), time=time, labels=self.labels)
            new_to.data[:] = out_arr
        else:
            new_to = TimeSeries(shape=out_arr.shape, time=time, labels=self.labels)
            new_to.data[:] = out_arr
            #return out_arr
        return new_to

    def vstack(self, new_data):
        """ Vertically stack data to a TimeSeries """
        data = NumpyArray(shape=(self.shape[0] + new_data.shape[0], new_data.shape[1]), parent=self)
        data[0:self.shape[0],:] = self.data
        data[self.shape[0]:,:] = new_data.data
        self.data = data
        #concat time index
        self._time = np.array(np.concatenate([self._time, new_data._time]), dtype=np.float64)


    def merge(self, tseries, fill=np.nan):
        """Merge multiple timeseries into one TimeSeries object

        Parameters
        ----------
        tseries: list of TimeSeries
            a list with time series that need to be merged with self
        fill:
            can be substituted with a float or np.float if nan is not desired for empty columns

        Returns
        -------
        A new TimeSeries object

        Example
        -------
        t1.merge([t2, t3])
        >>> TimeSeries([[1.56563280e+09, 5.22240746e-01, 1.93171146e+01, ...,
             5.38888889e-01, 9.68305908e+00, 2.98059849e+00],
            [1.56563370e+09, 5.33333333e-01, 1.95934302e+01, ...,
             4.55587586e-01, 9.99463997e+00, 3.07007493e+00],
            [1.56563460e+09, 5.22222222e-01, 1.94288035e+01, ...,
             4.55471427e-01, 7.87254330e+00, 2.96265267e+00],
            ...,
            [1.56571650e+09, 5.27796301e-01, 1.96337589e+01, ...,
             1.35845181e+00, 9.80688437e+00, 3.09112516e+00],
            [1.56571740e+09, 5.27705227e-01, 1.94194939e+01, ...,
             1.28194444e+00, 8.31783752e+00, 3.09129471e+00],
            [1.56571830e+09, 5.83333333e-01, 2.04479746e+01, ...,
             8.54166667e-01, 7.78839232e+00, 3.19808880e+00]])
        """
        # find out if timeindex are the same
        timeindex = self.time
        columns = self.shape[1]
        if not isinstance(tseries, list):
            tseries = [tseries]
        for t in tseries:
            if not isinstance(t, TimeSeries):
                raise (
                    NotImplementedError(
                        "A object of type {} cannot be combined with a TimeSeries".format(type(t))
                    )
                )
            if not np.array_equal(timeindex, t.time):
                # merge time index
                timeindex = self._merge_time_indicies(timeindex, t.time)
            columns += t.shape[1]

        # create new series
        new_t = TimeSeries(
            shape=(len(timeindex), columns), labels=self.labels.copy(), time=timeindex
        )
        if  fill != np.nan and isinstance(fill, (float, np.float64)):
            new_t[:] = fill

        row_mask = np.isin(timeindex, self.time)
        new_t.data[row_mask, 0 : self.data.shape[1]] = self.data

        # set the counter
        column_counter = self.shape[1]
        for t in tseries:
            t_columns = t.shape[1] + column_counter
            row_mask = np.isin(timeindex, t.time)
            new_t[row_mask, column_counter:t_columns] = t.data
            column_counter = t_columns
            for l in t.labels:
                new_t.labels.append(l.copy())
        return new_t

    def argfilter(self, clauses):
        """Filter the columns of data based on labels.

        Parameters
        ----------
        clauses : list
            clauses to filter on

        Returns
        -------
        a list of indexes for matching columns

        Example
        -------
        t.argfilter([{'hostname': 'host2'}])
        >>> [1,4,6]
        """
        stream_indexes = []
        for i, l in enumerate(self.labels.copy()):
            for k, v in clauses.items():
                if not isinstance(v, list):
                    v = [v]
                if k in l.keys() and l[k] in v:
                    stream_indexes.append(i)  # because first column is the time index
                    continue
        return stream_indexes

    def filter(self, clauses):
        """Filter the columns of data based on labels.

        Parameters
        ----------
        clauses : list
            clauses to filter on

        Returns
        -------
        a np.array view for matching columns

        Example
        -------
        t[{'hostname': 'host2'}]
        >>> TimeSeries([[1.56563280e+09, 5.22240746e-01, 1.93171146e+01, ...,
             5.38888889e-01, 9.68305908e+00, 2.98059849e+00],
            [1.56563370e+09, 5.33333333e-01, 1.95934302e+01, ...,
             4.55587586e-01, 9.99463997e+00, 3.07007493e+00],
            [1.56563460e+09, 5.22222222e-01, 1.94288035e+01, ...,
             4.55471427e-01, 7.87254330e+00, 2.96265267e+00],
            ...,
            [1.56571650e+09, 5.27796301e-01, 1.96337589e+01, ...,
             1.35845181e+00, 9.80688437e+00, 3.09112516e+00],
            [1.56571740e+09, 5.27705227e-01, 1.94194939e+01, ...,
             1.28194444e+00, 8.31783752e+00, 3.09129471e+00],
            [1.56571830e+09, 5.83333333e-01, 2.04479746e+01, ...,
             8.54166667e-01, 7.78839232e+00, 3.19808880e+00]])
        """
        return self[[self.argfilter(clauses)]]

    def group(self, group):
        """ Create a collection based on a grouping """
        masks = self._get_group_masks(group)
        collection = TimeSeriesCollection()
        collection.chunks = (TimeChunk(slice(None, None, None), mask) for mask in masks)
        collection.axis = 1
        collection.parent = self
        return collection

    def resample(self, period):
        """ Create a collection based on resampling by time """
        if isinstance(period, str):
            period = convert_string_to_seconds(period)
        collection = TimeSeriesCollection()
        collection.axis = 0
        collection.parent = self
        collection.collapse_index = True
        columns = [x for x in range(0,self.data.shape[1])]
        collection.chunks = (TimeChunk(slice(*self._get_slice_by_datetime(np.datetime64(int(t), "s"), np.timedelta64(period, "s"))), columns) for t in range(int(self.time[0]), int(self.time[-1]), period)) # pylint: disable-msg=too-many-function-args
        return collection

    def to_pandas(self):
        """ Return two pandas objects.  One for the label data and one for the measurements"""

        import pandas as pd

        columns = ["stream_" + str(x) for x in range(len(self.labels))]
        data = pd.DataFrame(data=self.data, columns=columns)
        data["time_index"] = pd.to_datetime(self.time, unit="s")
        data.set_index("time_index")
        label_columns = list(set(key for dic in self.labels for key in dic.keys()))
        label_data = [[row.get(key, None) for row in self.labels] for key in label_columns]
        labels = pd.DataFrame(columns=label_columns)
        for i, k in enumerate(label_columns):
            labels[k] = label_data[i]
        return (data, labels)

    def rolling_window(self, size, stepsize=1):
        """ Create a collection based on a rolling window """
        collection = TimeSeriesCollection()
        collection.axis = 0
        collection.parent = self
        collection.collapse_index = True
        init_data = TimeSeries(shape=(size-1, self.shape[1]), labels=self.labels, time=self._time[:size-1])
        init_data[:] = np.nan
        collection.init_data = init_data
        collection.collapse_time_index = -1
        collection.chunks = (TimeChunk(slice(i, i+size), slice(None, None, None)) for i in range(0, len(self)-size+1, stepsize))
        return collection

    def fillnan(self, method):
        """ Replace np.nan with a number, or foward/back fill """
        if isinstance(method, str) and method in ['bfill', 'ffill']:
            modifier = -1
            if method == 'bfill':
                modifier = 1
            for i in range(self.data.shape[1]):
                while True:
                    idx = np.argwhere(np.isnan(self.data[:,i]))
                    if len(idx) == 0:
                        break
                    row, col = idx[-1]
                    col = i
                    if 0 < (row + modifier) < len(self.data):
                        self.data[row,col] = self.data[row+modifier,col]
                    else:
                        self.data[row,col] = 0
        elif isinstance(method, numbers.Number):
            self.data[np.isnan(self.data).astype(bool)] = method #need as bool because all values are floats
        else:
            raise ValueError("Method should be 'bfill','ffill', or a number")


    def bfill(self):
        """ Back fill np.nan """
        return self.fillnan('bfill')

    def ffill(self):
        """ Forward fill np.nan """
        return self.fillnan('ffill')

    def fill(self, value):
        """ fill with a number np.nan """
        return self.fillnan(value)

    def diff(self, *args, **kwargs):
        """ Find the n difference for rows """
        if 'axis' not in kwargs:
            kwargs['axis'] = 0
        data = np.diff(self, *args, **kwargs)
        new_t = TimeSeries(shape=data.shape, time=self.time[-data.shape[0]:])
        new_t.labels = self.labels
        new_t[:] = data
        return new_t

    def copy(self):
        """Override the copy method to include the labels

        Parameters
        ----------
        None

        Returns
        -------
        A new TimeSeries object

        Example
        -------
        new_t.copy()
        >>> TimeSeries([[1.56563280e+09, 5.22240746e-01, 1.93171146e+01, ...,
             5.38888889e-01, 9.68305908e+00, 2.98059849e+00],
            [1.56563370e+09, 5.33333333e-01, 1.95934302e+01, ...,
             4.55587586e-01, 9.99463997e+00, 3.07007493e+00],
            [1.56563460e+09, 5.22222222e-01, 1.94288035e+01, ...,
             4.55471427e-01, 7.87254330e+00, 2.96265267e+00],
            ...,
            [1.56571650e+09, 5.27796301e-01, 1.96337589e+01, ...,
             1.35845181e+00, 9.80688437e+00, 3.09112516e+00],
            [1.56571740e+09, 5.27705227e-01, 1.94194939e+01, ...,
             1.28194444e+00, 8.31783752e+00, 3.09129471e+00],
            [1.56571830e+09, 5.83333333e-01, 2.04479746e+01, ...,
             8.54166667e-01, 7.78839232e+00, 3.19808880e+00]])
        """
        ts = TimeSeries(shape=self.shape, labels=self.labels.copy(), time=self._time)
        ts.data = self.data.copy()
        return ts


def delegate(cls, attr_name, method_name):
    def delegated(self, *vargs, **kwargs):
        a = getattr(self, attr_name)
        m = getattr(a, method_name)
        return m(*vargs, **kwargs)
    setattr(cls, method_name, delegated)

dunder_funcs = ['truediv', 'add', 'sub', 'mul', 'floordiv', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or', 'iadd', 'isub', 'imul', 'idiv', 'ifloordiv', 'imod', 'ipow', 'ilshift', 'irshift', 'iand', 'ixor', 'radd', 'rsub', 'rmul', 'rdiv', 'rfloordiv', 'rmod', 'rpow', 'rlshift', 'rand', 'rxor', 'ror', 'lt', 'le', 'ne', 'eq', 'gt', 'ge','len']

# proxy the magic methods to the numpy array
for name in dunder_funcs:
    name = f"__{name}__"
    delegate(TimeSeries, 'data', name)