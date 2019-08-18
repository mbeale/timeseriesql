import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, MonthLocator, YearLocator, MinuteLocator, DateFormatter

class TimeSeries(np.ndarray):
    """ 
    TimeSeries
    ----------

    This class extends the numpy array using a column that contains a timestamp
    index and X number of columns that represent labelled data.  Timeindexing and 
    filtering/grouping by label(s) is also added.
    """
    
    def __new__(self, shape, labels=None):
        """Creates a new array based on the shape"""
        obj = np.empty(shape, dtype=np.float64).view(self)
        obj[:] = np.nan
        if isinstance(labels, list):
            obj.labels = [l for l in labels]
        else:
            obj.labels = [labels]
        return obj

    def __array_finalize__(self, obj):
        """Used to add labels to the object"""
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)
        
    def __getitem__(self, items):
        """Add functionality for filtering on labels and datetime objects"""
        # if a label filter, return view for the correct streams
        try: 
            return super(TimeSeries, self).__getitem__(items)
        except IndexError as e:
            if isinstance(items, np.datetime64):
                start, stop = self._get_slice_by_datetime(items)
                if (start, stop) == (None, None):
                    raise e
                return super(TimeSeries, self).__getitem__(start)
            l = len(items)
            if l != 1:
                raise e
            if l == 1:
                filters = [0] + self.argfilter(items)
                if len(filters) == 1:
                    return None #maybe should return something else?
                return super(TimeSeries, self).__getitem__((slice(None), filters))
        except TypeError as e:
            #build new slice
            if isinstance(items, slice):
                start, stop = self._get_slice_by_datetime(items.start, items.stop)
                if (start, stop) == (None, None):
                    raise e
                return super(TimeSeries, self).__getitem__(slice(start, stop))
            raise e
                    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override ufunc by only processing the data with ufuncs while ignoring the time index"""
        args = []
        labels = []

        #clean inputs
        for input in inputs:
            #check if time series have same timeindex
            if isinstance(input, TimeSeries):
                args.append(input.get_data)
                labels.append(input.labels)
            else:
                args.append(input)
        #clean outputs
        if 'out' in kwargs:
            new_out = []
            for o in kwargs['out']:
                if isinstance(o, TimeSeries):
                    new_out.append(o.get_data)
                else:
                    new_out.append(o)
            kwargs['out'] = tuple(new_out)
    
        # then just call the parent
        results = super(TimeSeries, self).__array_ufunc__(ufunc, method,
                                                 *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        
        columns = 2 # 2 = timeindex plus the one column
        if len(results.shape) == 1:  #reduced to one column
            results = results.reshape((results.shape[0], 1))
        else:
            columns = results.shape[1] + 1

        # create an empty time series
        t = TimeSeries((len(self), columns), labels=labels[0]) #should merge the labels
        t[:] = np.hstack([self[:,0].reshape(len(self), 1), results])
        return t

    @property
    def get_data(self):
        """Return the data portion of the object"""
        return self[:,1:].view(np.ndarray)
    
    @property
    def get_time_index(self):
        """Return only the TimeSeries index"""
        return self[:,0].view(np.ndarray)

    def _get_group_masks(self, group):
        """Return a mask for each label"""
        if not isinstance(group, list):
            group = [group]
        groupings = {}
        for i,stream in enumerate(self.labels):
            key = ""
            for g in group:
                if g in stream:
                    key += stream[g]
            if key in groupings:
                groupings[key].append(i+1)
            else:
                groupings[key] = [0, i+1]
        return [x for x in groupings.values()]

    def _get_slice_by_datetime(self, start_date=None, stop_date=None):
        """Return the indicies for a TimeSeries object searching by a datetime object"""
        start, stop = (None, None)
        if isinstance(start_date, np.timedelta64):
            start_date = np.datetime64(int(self.get_time_index[0]),'s') + start_date
        if isinstance(start_date, np.datetime64):
            start = np.argwhere(self.get_time_index >= start_date.astype(float))
            if len(start) > 0:
                start = start[0][0]
            else:
                raise IndexError("No timestamps exist at or after %s" % start_date)
        if isinstance(stop_date, np.timedelta64):
            if start_date:
                stop_date = start_date + stop_date
            else:
                stop_date = np.datetime64(int(self.get_time_index[-1]),'s') - stop_date
        if isinstance(stop_date, np.datetime64):
            stop = np.argwhere(self.get_time_index >= stop_date.astype(float))
            if len(stop) > 0:
                stop = stop[0][0]
            else:
                raise IndexError("No timestamps exist before or at %s" % stop_date)
        return (start, stop)

    def _merge_time_indicies(self, t1, t2):
        """Merge time indicies to crate an a new index of the union of the two indicies"""
        res = sorted(set(t1).union(set(t2)))
        return np.array(res)

    def _get_unique_keys(self):
        """Get list of unique keys"""
        labels = self.labels[0].copy()
        for l in self.labels[1:]:
            for k in labels.copy().keys():
                if k not in l or l[k] != labels[k]:
                    del labels[k]
        return labels

    def _generate_title(self):
        """Generate a title based on common tags"""
        #get common labels
        labels = self._get_unique_keys()
        if len(labels) > 0:
            return '.'.join([str(v) for k,v in labels.items()])
        return 'TimeSeries'

    def _generate_labels(self):
        """Generate labels for legend creation"""
        labels = []
        exclude_labels = self._get_unique_keys()
        for l in self.labels:
            labels.append('.'.join([str(v) for k,v in l.items() if k not in exclude_labels]))
        return labels

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
        copy = super(TimeSeries, self).copy()
        ts = TimeSeries(copy.shape, labels=self.labels)
        ts[:] = copy[:]
        return ts


    def group(self, group, ufunc=np.add):
        """Merge multiple timeseries into one TimeSeries object

        Parameters
        ----------
        group: list of labels
            a list of labels to group the streams by
        ufunc: ufunc 
            a ufunc to use to combined grouped streams

        Returns
        -------
        A new TimeSeries object

        Example
        -------
        new_t.group(['hostname','name'])
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
        masks = self._get_group_masks(group)
        new_t = TimeSeries((len(self), len(masks) + 1))
        new_t[:,0] = self.get_time_index
        for i, mask in enumerate(masks):
            if isinstance(ufunc, np.ufunc):
                new_t[:, [i+1]] = ufunc.reduce(self[:,mask], axis=1).get_data
            else:
                new_t[:, [i+1]] = ufunc(self[:,mask]).get_data
        return new_t

    def merge(self, tseries):
        """Merge multiple timeseries into one TimeSeries object

        Parameters
        ----------
        tseries: list of TimeSeries
            a list with time series that need to be merged with self

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
        #find out if timeindex are the same
        timeindex = self.get_time_index
        columns = self.shape[1]
        if not isinstance(tseries, list):
            tseries = [tseries]
        for t in tseries:
            if not isinstance(t, TimeSeries): 
                raise(NotImplementedError(
                    "A object of type {} cannot be combined with a TimeSeries".format(type(t))
                    )
                )
            if not np.array_equal(timeindex, t.get_time_index):
                #merge time index
                timeindex = self._merge_time_indicies(timeindex, t.get_time_index)
            columns += t.shape[1] - 1
        
        #create new series
        new_t = TimeSeries((len(timeindex), columns), labels=self.labels)

        #load data from self
        new_t[:,0] = timeindex

        row_mask = np.isin(timeindex, self.get_time_index)
        new_t.get_data[row_mask,0:self.get_data.shape[1]] = self.get_data

        #set the counter
        column_counter = self.shape[1]
        for t in tseries:
            t_columns = t.shape[1] - 1 + column_counter
            row_mask = np.isin(timeindex, t.get_time_index)
            new_t[row_mask,column_counter:t_columns] = t.get_data
            column_counter = t_columns
            for l in t.labels:
                new_t.labels.append(l.copy())
        return new_t

    def plot(self, ax=None, legend=True, **kwargs):
        """Plot charts using sane time series defaults with Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            an Axes to plot against.  One will be created if not included
        legend: Boolean
            Decision to generate a legend

        Returns
        -------
        None

        Example
        -------
        cpu.plot()
        >>>
        """

        date_index = [np.datetime64(int(x),'s') for x in np.nditer(self.get_time_index)]
        if not ax:
            fig = plt.figure(1)
            fig.autofmt_xdate()
            ax = fig.add_subplot(111)

        window = (date_index[-1] - date_index[0]).astype(int) 

        xlabel = ''
        if window <= 3600:
            minor_locator = MinuteLocator()
            minor_formatter = DateFormatter('%M')
            major_locator = HourLocator()
            major_formatter = DateFormatter('\n%Y-%m-%d %H:%M')
            xlabel = 'Minute'
        elif window <= 86400:
            minor_locator = HourLocator()
            minor_formatter = DateFormatter('%H:%M')
            major_locator = DayLocator()
            major_formatter = DateFormatter('\n%Y-%m-%d')
            xlabel = 'Hour'
        elif window <= (7 * 86400):
            minor_locator = HourLocator(interval=6)
            minor_formatter = DateFormatter('%H:%M')
            major_locator = DayLocator()
            major_formatter = DateFormatter('\n%Y-%m-%d')
            xlabel = 'Hour'
        elif window <= (60 * 86400):
            interval = 1
            if len(date_index) > 30:
                interval = 2
            minor_locator = DayLocator(interval=interval)
            minor_formatter = DateFormatter('%m-%d')
            major_locator = YearLocator()
            major_formatter = DateFormatter('\n%Y')
            xlabel = 'Day'
        else:
            minor_locator = MonthLocator()
            minor_formatter = DateFormatter('%B')
            major_locator = YearLocator()
            major_formatter = DateFormatter('\n%Y')
            xlabel = 'Month'

        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_minor_formatter(minor_formatter)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
        
        plt.title(self._generate_title(), fontsize=18)
        plt.xlabel = xlabel
        lines = ax.plot(date_index, self.get_data, **kwargs)
        if legend:
            plt.legend(lines,self._generate_labels(), title='Streams')


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
        for i,l in enumerate(self.labels):
            for k, v in clauses.items():
                if not isinstance(v, list):
                    v = [v]
                if k in l.keys() and l[k] in v:
                    stream_indexes.append(i+1) #because first column is the time index
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

    def resample(self):
        #reduce and expand
        pass