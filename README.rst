A Pythonic query language for time series data
----------------------------------------------

**ALPHA Software.  Not ready for production usage**

This library attempts to use generators in python as way to query various time series DBs. 

Examples:
---------

  #gets metric named test and returns the mean value

  q = Query(x.mean for x in "test")
  
  #gets metric named test where label1 is equal to 'prod'
  
  q = Query(x for x in "test" if x.label1 == 'prod') 
  
  # get 2 metrics and find the difference between max and min of those metrics grouped by the 'tag1' label
  
  q = Query(x - y for x,y in Query((x.max for x in "metric1"), (x.min for x in "metric2")).by('tag1'))
  
  #get streams for metric test from now-3600 seconds to now-1800 seconds with a resolution of 300 seconds
  
  q = Query(x.mean for x in "test")[3600:1800:300]

All queries return a numpy array with a time index column and one column per stream of data returned.  the
labels can be accessed via the `label` attribute.  

All ufuncs can be used against the data without modifying the time index columns

Examples:
---------

  #grouping by mean
  
  reduced = t.group('env', lambda x: np.add.reduce(x, axis=1) / (x.shape[1] - 1))

You can see the tests for more inspiration.

License
-------

This project is Copyright (c) Michael Beale and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


Contributing
------------

If you would like to write additional backends for various time series databases, 
please reach out first.
