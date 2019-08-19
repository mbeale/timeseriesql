import unittest
import numpy as np
import copy
import time
import math
from timeseriesql.timeseries import TimeSeries

class TestTimeSeries(unittest.TestCase):

    def setUp(self):
        now = math.floor(int(time.time()) / 3600) * 3600
        timeindex = np.array([np.float64(now + (x * 60)) for x in range(10)], dtype=np.float64)
    
        t1 = TimeSeries((10,2), labels={'name': 'metric1', 'env': 'prod', 'hostname': 'host1'})
        t1[:] = [[i,v] for i, v in zip(timeindex, range(10))]
        t2 = TimeSeries((10,2), labels={'name': 'metric1', 'env': 'prod', 'hostname': 'host2'})
        t2[:] = [[i,v] for i, v in zip(timeindex, range(10))]
        t3 = TimeSeries((10,2), labels={'name': 'metric1', 'env': 'prod', 'hostname': 'host3'})
        t3[:] = [[i,v] for i, v in zip(timeindex, range(10))]

        self.timeindex = timeindex
        self.basic_timeseries = (t1,t2,t3)

    def test_init_array(self):
        shape = (8, 9)
        t = TimeSeries(shape)
        self.assertEqual(shape, t.shape)

    def test_add_stream_full(self):

        t1,t2,t3 = self.basic_timeseries

        new_t = t1.merge([t2, t3])
        self.assertTrue(np.array_equal(self.timeindex, new_t.get_time_index))
        self.assertEqual(new_t[2,1], 2.)
        self.assertEqual(new_t[2,2], 2.)
        self.assertEqual(new_t[2,3], 2.)

    def test_reduce_add_internal(self):
        t1,t2,t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        reduced_t = np.add.reduce(new_t, 1)

        self.assertEqual(reduced_t[2,1], 6)
        self.assertEqual(reduced_t[3,1], 9)
        self.assertEqual(reduced_t[4,1], 12)

    def test_add_scalar(self):
        t1,t2,t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        rtn_new = new_t + 5
        self.assertEqual(rtn_new[0,0], t2[0,0])
        self.assertEqual(rtn_new[0,1], 5.0)
        self.assertEqual(rtn_new[1,2], 6.0)
        self.assertEqual(rtn_new[2,3], 7.0)
        self.assertEqual(rtn_new[3,2], 8.0)

        new_t += 5

        self.assertEqual(new_t[0,1], 5.0)
        self.assertEqual(new_t[1,2], 6.0)
        self.assertEqual(new_t[2,3], 7.0)
        self.assertEqual(new_t[3,2], 8.0)

    def test_combine_two_series_with_add(self):
        t1,t2,t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        new_t += copy.deepcopy(new_t)

        self.assertEqual(new_t.shape, (10,4))
        self.assertEqual(new_t[3,2], 6)
        self.assertEqual(new_t[4,1], 8)

    def test_add_unequal_series(self):
        t1,t2,t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        #create a larger time series, starts after
        new_index = np.hstack([self.timeindex, [x for x in range(int(self.timeindex[-1]) + 60,int(self.timeindex[-1])+240, 60)]])
    
        t1 = TimeSeries((13,2), labels={'name': 'metric2', 'env': 'prod', 'hostname': 'host1'})
        t1[:] = [[i,v] for i, v in zip(new_index, range(13))]
        t2 = TimeSeries((13,2), labels={'name': 'metric2', 'env': 'prod', 'hostname': 'host2'})
        t2[:] = [[i,v] for i, v in zip(new_index, range(13))]
        t3 = TimeSeries((13,2), labels={'name': 'metric2', 'env': 'prod', 'hostname': 'host3'})
        t3[:] = [[i,v] for i, v in zip(new_index, range(13))]
        larger_t = t1.merge([t2, t3])

        merged_t = new_t.merge([larger_t])

        self.assertEqual(merged_t.shape, (13,7))
        self.assertEqual(merged_t[12,5], 12.0)
        self.assertTrue(np.isnan(merged_t[12,2]))

        #create a smaller time series, starts before
        new_index = self.timeindex[2:8]
        t1 = TimeSeries((6,2), labels={'name': 'metric2', 'env': 'prod', 'hostname': 'host1'})
        t1[:] = [[i,v] for i, v in zip(new_index, range(6))]
        t2 = TimeSeries((6,2), labels={'name': 'metric2', 'env': 'prod', 'hostname': 'host2'})
        t2[:] = [[i,v] for i, v in zip(new_index, range(6))]
        t3 = TimeSeries((6,2), labels={'name': 'metric2', 'env': 'prod', 'hostname': 'host3'})
        t3[:] = [[i,v] for i, v in zip(new_index, range(6))]
        smaller_t = t1.merge([t2, t3])

        merged_t = new_t.merge([smaller_t])

        self.assertEqual(merged_t.shape, (10,7))
        self.assertEqual(merged_t[5,2], 5.0)
        self.assertTrue(np.isnan(merged_t[9,5]))
        

    def test_merge_time_index(self):
        t1,_,_ = self.basic_timeseries
        new_index = np.hstack([self.timeindex, [x for x in range(int(self.timeindex[-1]) + 60,int(self.timeindex[-1])+240, 60)]])

        rtn1 = t1._merge_time_indicies(self.timeindex, new_index)
        self.assertTrue(np.array_equal(rtn1, new_index))
        self.assertTrue(np.array_equal(t1._merge_time_indicies(self.timeindex[[0,1,3,5,7]], new_index), new_index))

    def test_merge_success(self):
        t1,t2,t3 = self.basic_timeseries
        values = [t1, t2, t3, t2, t2, t3]
        new_t = t1.merge(values[1:])
        for i, t in enumerate(values):
            self.assertTrue(np.array_equal(new_t[:,[i+1]], t.get_data))
    
    def test_group_reduce(self):
        t1,t2,t3 = self.basic_timeseries

        #one label
        new_t = t1.merge([t2, t3, t2, t2, t3])
        reduced = new_t.group('hostname')
        host2 = new_t[:,[0,2]] + new_t[:,[0,4]] + new_t[:,[0,5]]
        self.assertTrue(np.array_equal(reduced[:,[0,2]], host2))

        #multiple labels
        new_t.labels[1]['name'] = 'metric2'
        reduced = new_t.group(['hostname','name'])
        self.assertEqual(reduced.shape[1], 5)
        host2 = new_t[:,[0,4]] + new_t[:,[0,5]]
        self.assertTrue(np.array_equal(reduced[:,[0,2]], new_t[:,[0,2]]))
        self.assertTrue(np.array_equal(reduced[:,[0,4]], host2))

        #non reduce ufunc (mean)
        avg_1,avg_2,avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])
        reduced = new_t.group('env', lambda x: np.add.reduce(x, axis=1) / (x.shape[1] - 1))
        self.assertTrue(np.array_equal(reduced[:,[0,1]], avg_2))


    def test_index_by_label(self):
        t1,t2,t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        self.assertTrue(np.array_equal(new_t[{'hostname': 'host2'}], new_t[:,[0,2]]))
        self.assertTrue(np.array_equal(new_t[{'hostname': ['host2', 'host3']}], new_t[:,[0,2,3]]))
        self.assertIsNone(new_t[{'hostname': 'does_not_exist'}])
        

    def test_index_by_datetime(self):

        t1,t2,t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        beg = np.datetime64(int(self.timeindex[2]),'s') #pylint: disable-msg=too-many-function-args
        end = np.datetime64(int(self.timeindex[5]),'s') #pylint: disable-msg=too-many-function-args

        #datetime range
        self.assertTrue(np.array_equal(new_t[beg:end], new_t[2:5]))

        #specific time
        self.assertTrue(np.array_equal(new_t[beg], new_t[2]))

        #set start time to the end
        self.assertTrue(np.array_equal(new_t[beg:], new_t[2:]))

        #beginning to set end time
        self.assertTrue(np.array_equal(new_t[:end], new_t[:5]))

        #get 3 minutes of data from start point
        self.assertTrue(np.array_equal(new_t[beg:np.timedelta64(3, 'm')], new_t[2:5])) #pylint: disable-msg=too-many-function-args

        #offset 3 minutes of data
        self.assertTrue(np.array_equal(new_t[np.timedelta64(3, 'm'):], new_t[3:])) #pylint: disable-msg=too-many-function-args

        #end 3 minutes before ending
        self.assertTrue(np.array_equal(new_t[:np.timedelta64(3, 'm')], new_t[:6])) #pylint: disable-msg=too-many-function-args