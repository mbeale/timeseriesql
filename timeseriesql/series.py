from dataclasses import dataclass, field
from typing import Dict, Type, Tuple
import numpy as np
import datetime


@dataclass
class Series:
    """ This class holds a collection of streams for a particular metric """
    metadata: Dict[str, any] = field(default_factory=dict)
    measurements: Type[np.ndarray] = field(default=np.array([]))
    timestamps: Type[np.ndarray] = field(default=np.array([]))    
    streams: Dict[str, any] = field(default_factory=dict)
    dtype: Tuple[str, str] = field(default=tuple([]))
    columns: int = 0

    def init_array(self, columns:int):
        """ initialize the array and create the dtype """
        self.columns = columns
        self.measurements = np.array([],dtype=np.float32)

    def get_next_stream_key(self):
        return len(self.streams)

    def get_values(self):
        return self.measurements

   
    def add_stream(self, timeseries, values, tags):
        key:int = self.get_next_stream_key()
        sorted_tags = {key:tags[key] for key in sorted(tags.keys())}
        self.streams.update({key: sorted_tags})
        if self.timestamps.size == 0:
            self.timestamps = np.array(timeseries)
            self.measurements = np.array([[np.nan for x in np.arange(self.columns)] for t in timeseries]).reshape(len(timeseries), self.columns)
        elif not np.array_equal(self.timestamps, timeseries):
            new_series = Series(metadata={"name": self.metadata['name']})
            new_series.init_array(self.columns)
            new_series.streams = self.streams
            current_ndx = 0
            new_ndx = 0
            c_copy_ndx = 0
            while current_ndx is not None or new_ndx is not None:
                if current_ndx and current_ndx >= len(self.timestamps):
                    new_series.measurements = np.append(new_series.measurements, self.measurements[c_copy_ndx:])
                    current_ndx = None
                if new_ndx and new_ndx >= len(timeseries):
                    new_ndx = None
                if current_ndx == None and new_ndx != None:
                    new_series.measurements = np.append(new_series.measurements, 
                        np.array([np.nan for x in np.arange(self.columns)]))
                    new_ndx += 1
                    continue
                elif current_ndx != None and new_ndx == None:
                    values = np.hstack([values[:new_ndx], [np.nan]])
                    current_ndx += 1
                    continue
                elif current_ndx == None and new_ndx == None:
                    continue
                if self.timestamps[current_ndx] == timeseries[new_ndx]:
                    new_ndx += 1
                    current_ndx += 1
                elif self.timestamps[current_ndx] < timeseries[new_ndx]:
                    values = np.hstack([values[:new_ndx], [self.timestamps[current_ndx]], values[new_ndx:]])
                    current_ndx += 1
                else:
                    if c_copy_ndx != current_ndx:
                        new_series.measurements = np.append(new_series.measurements, self.measurements[c_copy_ndx:current_ndx])
                        c_copy_ndx = current_ndx
                    new_series.measurements = np.append(new_series.measurements, np.array([np.nan for x in np.arange(self.columns)]))
                    new_ndx += 1
            self.replace_data(new_series)
            del(new_series)
        self.measurements[:,key] = values

    """
    def add_stream(self, timeseries, values, tags):
        key:int = self.get_next_stream_key()
        sorted_tags = {key:tags[key] for key in sorted(tags.keys())}
        self.streams.update({key: sorted_tags})
        if self.timestamps.size == 0:
            self.timestamps = np.array(timeseries)
            self.measurements = np.array([[np.nan for x in np.arange(self.columns)] for t in timeseries]).reshape(len(timeseries), self.columns)
        elif not np.array_equal(self.timestamps, timeseries):
            #need to resize/reorder
        self.measurements[:,key] = values 
     """   

    def replace_data(self, new_series):
        self.dtype = new_series.dtype
        sh = new_series.measurements.shape
        if len(sh) == 1:
            self.measurements = new_series.measurements.reshape(int(sh[0] / self.columns), self.columns)
        else:
            self.measurements = new_series.measurements
        self.timestamps = new_series.timestamps
        self.columns = new_series.columns
        self.streams = new_series.streams
        self.metadata = new_series.metadata

    def combine_stream_metadata(self, metadata):
        if len(metadata) == 1:
            return metadata[0]
        current = metadata[0]
        for m in metadata[1:]:
            current = {k:current[k] for k in current.keys() if k in m.keys() if current[k] == m[k]}
        return current

    def group_streams(self, group_by):
        if isinstance(group_by, str):
            group_by = [group_by]
        elif group_by == False:
            return {"all":[x for x in self.streams.keys()]}
        streams_keys = {}
        for stream in self.streams:
            key = ''.join(["{}:{}".format(x, self.streams[stream][x]) for x in sorted(self.streams[stream].keys()) if x in group_by or group_by[0] == "*"])
            if key in streams_keys:
                streams_keys[key].append(stream)
            else:
                streams_keys[key] = [stream]
        return streams_keys
        

    def group_reduce_streams(self, group_by, group_by_func):
        new_series = Series(metadata={"name": self.metadata['name']})
        stream_keys = self.group_streams(group_by)
        new_series.init_array(len(stream_keys))
        for key in stream_keys:
            new_series.add_stream(self.timestamps, group_by_func([self.measurements[:,x] for x in stream_keys[key]]), self.combine_stream_metadata([self.streams[x] for x in stream_keys[key]]))
        self.replace_data(new_series)
        del(new_series)

    def diminish(self, func):
        pass

          

    def add(self, value, combine=True, group_by='*', inplace=False):
        """ add traditional add method """
        if isinstance(value, (float, int)):
            if inplace:
                self.measurements += value
            else:
                return self.measurements + value
        elif isinstance(value, Series):
            metric_name = self.metadata['name'] if self.metadata['name'] == value.metadata['name']  else ''
            new_series = Series(metadata={"name": metric_name})
            columns:int = self.columns + value.columns
            new_series.init_array(columns)
            for series in [self, value]:
                for s, m in zip(series.measurements.T, series.streams):
                    new_series.add_stream(series.timestamps, s, series.streams[m])
            self.replace_data(new_series)
            del(new_series)
            if combine:
                self.group_reduce_streams(group_by, self.reduce_add)
 
        else:
            raise(NotImplementedError(
                "A object of type {} cannot be added to a Series".format(type(value))
                )
            )
        return self.measurements.view()

    def reduce_add(self, values):
        """ reduce using numpy add and axis=0 """
        return np.nansum(values, axis=0)

    def add_(self):
        """ reducing by adding of values """
        self.group_reduce_streams(False, self.reduce_add)
        return self.measurements.view()

    #magic methods

    def __getitem__(self, items):
        # TODO:// Lambda filters
        if isinstance(items, tuple):
            rows = items[0]
            cols = items[1]
        else:
            rows = items
            cols = None
        if isinstance(rows, (int, slice)):
            row_selector = rows
        elif isinstance(rows, list) and isinstance(rows[0], (datetime.datetime, datetime.date)):
            start_index = (self.timestamps >= int(rows[0].timestamp())).nonzero()[0][0]
            if len(rows) == 2:
                row_selector = slice(start_index, (self.timestamps >= int(rows[1].timestamp())).nonzero()[0][0])
            else:
                row_selector = start_index
        elif isinstance(rows, np.ndarray) and rows.dtype == np.dtype('bool'):
            row_selector = rows
        else:
            raise ValueError("Only integers, slices, list of date(time)s, and boolean arrays are valid")
        if cols:
            if isinstance(cols, int):
                col_selector = cols
            elif isinstance(cols, slice):
                col_selector = [x for x in self.streams.keys()][cols]
            elif isinstance(cols, np.ndarray) and rows.dtype == np.dtype('bool'):
                col_selector = cols
            else:
                raise ValueError("Only integers, slices, and boolean arrays are valid")
            return self.measurements[row_selector,col_selector]
        return self.measurements[row_selector]

    def __add__(self, value):
        """ magic method """
        return self.add(value)

    def __iadd__(self, value):
        return self.add(value, inplace=True)
        

    def __len__(self):
        return len(self.measurements['timestamp'])
