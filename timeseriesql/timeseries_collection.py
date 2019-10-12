import numpy as np


class TimeSeriesCollection:
    """ A grouping of rows and columns for a TimeSeries 
    
    Attributes
    ----------

    data : np.array
        The place holder for array to be returned
    axis : int
        The axis for the function to be applied on
    collapse_index : bool
        Should the rows be squashed into 1 row
    collapse_time_index : int
        Which row index should be used.  0 for first, -1 for last
    chunks : [TimeChunk]
        A list of chunks to iterate over
    parent : TimeSeries
        The parent of the collection
    init_data : np.array
        If there is any initial data to start with
    """

    data = None
    axis = 0
    collapse_index = False
    collapse_time_index = 0
    chunks = []
    parent = None
    init_data = None

    def __getattr__(self, attr_name):
        """ Return a method for process each chunk """
        from .timeseries import TimeSeries

        def method(*args, **kwargs):
            if attr_name in dir(np) and callable(getattr(np, attr_name)):
                rtn_time_series = self.init_data
                for chunk in self.chunks:
                    if "axis" not in kwargs:
                        kwargs["axis"] = self.axis
                    if isinstance(getattr(np, attr_name), np.ufunc):
                        data = getattr(np, attr_name).reduce(
                            self.parent[chunk.row_mask][:, chunk.col_mask], *args, **kwargs
                        )
                    else:
                        data = getattr(np, attr_name)(
                            self.parent[chunk.row_mask][:, chunk.col_mask], *args, **kwargs
                        )
                    if not data.shape:  # not an array
                        return data
                    if len(data.shape) == 1:
                        if kwargs["axis"] == 0:
                            data = data.reshape((1, data.shape[0]))
                    elif data.shape[1] == 1 and data.shape[0] != 1:
                        if kwargs["axis"] == 0:
                            data = data.reshape((data.shape[1], data.shape[0]))

                    row_count = data.shape[0]
                    if len(data.shape) == 1:
                        col_count = 1
                    else:
                        col_count = data.shape[1]

                    time_index = self.parent.time[chunk.row_mask]
                    if self.collapse_index and row_count > 0:
                        time_index = [time_index[self.collapse_time_index]]
                        row_count = 1
                    new_ts = TimeSeries(
                        shape=(row_count, col_count),
                        time=time_index,
                        labels=self.parent._least_common_labels(chunk.col_mask),
                    )
                    new_ts[:] = data
                    if rtn_time_series is None:
                        rtn_time_series = new_ts
                    else:
                        if self.collapse_index:  # need to vstack this
                            rtn_time_series.vstack(new_ts)
                        else:
                            rtn_time_series = rtn_time_series.merge([new_ts])
                return rtn_time_series
            else:
                raise AttributeError

        return method
