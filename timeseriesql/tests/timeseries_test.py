import unittest
import numpy as np
import copy
import time
import math
from timeseriesql.timeseries import TimeSeries
import matplotlib.pyplot as plt


class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        now = math.floor(int(time.time()) / 3600) * 3600
        timeindex = np.array([np.float64(now + (x * 60)) for x in range(10)], dtype=np.float64)

        t1 = TimeSeries(
            shape=(10, 1),
            labels={"name": "metric1", "env": "prod", "hostname": "host1"},
            time=timeindex,
        )
        t1[:] = np.array([i for i in range(10)]).reshape(10, 1)
        t2 = TimeSeries(
            shape=(10, 1),
            labels={"name": "metric1", "env": "prod", "hostname": "host2"},
            time=timeindex,
        )
        t2[:] = np.array([i for i in range(10)]).reshape(10, 1)
        t3 = TimeSeries(
            shape=(10, 1),
            labels={"name": "metric1", "env": "prod", "hostname": "host3"},
            time=timeindex,
        )
        t3[:] = np.array([i for i in range(10)]).reshape(10, 1)

        self.timeindex = timeindex
        self.basic_timeseries = (t1, t2, t3)

    def test_init_array(self):
        shape = (8, 9)
        t = TimeSeries(shape=shape)
        self.assertEqual(shape, t.shape)

    def test_add_stream_full(self):

        t1, t2, t3 = self.basic_timeseries

        new_t = t1.merge([t2, t3])
        self.assertTrue(np.array_equal(self.timeindex, new_t.time))
        self.assertEqual(new_t[2, 0], 2.0)
        self.assertEqual(new_t[2, 1], 2.0)
        self.assertEqual(new_t[2, 2], 2.0)

    def test_reduce_add_internal(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        reduced_t = np.add.reduce(new_t, 1)

        self.assertEqual(reduced_t[2, 0], 6)
        self.assertEqual(reduced_t[3, 0], 9)
        self.assertEqual(reduced_t[4, 0], 12)

    def test_add_scalar(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        rtn_new = new_t + 5
        self.assertEqual(rtn_new.time[0], t2.time[0])
        self.assertEqual(rtn_new[0, 0], 5.0)
        self.assertEqual(rtn_new[1, 1], 6.0)
        self.assertEqual(rtn_new[2, 2], 7.0)
        self.assertEqual(rtn_new[3, 1], 8.0)

        new_t += 5

        self.assertEqual(new_t[0, 0], 5.0)
        self.assertEqual(new_t[1, 1], 6.0)
        self.assertEqual(new_t[2, 2], 7.0)
        self.assertEqual(new_t[3, 1], 8.0)

    def test_combine_two_series_with_add(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        new_t += copy.deepcopy(new_t)

        self.assertEqual(new_t.shape, (10, 3))
        self.assertEqual(new_t[3, 1], 6)
        self.assertEqual(new_t[4, 0], 8)

    def test_add_unequal_series(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        # create a larger time series, starts after
        new_index = np.hstack(
            [
                self.timeindex,
                [x for x in range(int(self.timeindex[-1]) + 60, int(self.timeindex[-1]) + 240, 60)],
            ]
        )

        t1 = TimeSeries(
            shape=(13, 1),
            labels={"name": "metric2", "env": "prod", "hostname": "host1"},
            time=new_index,
        )
        t1[:] = np.array([i for i in range(13)]).reshape(13, 1)
        t2 = TimeSeries(
            shape=(13, 1),
            labels={"name": "metric2", "env": "prod", "hostname": "host2"},
            time=new_index,
        )
        t2[:] = np.array([i for i in range(13)]).reshape(13, 1)
        t3 = TimeSeries(
            shape=(13, 1),
            labels={"name": "metric2", "env": "prod", "hostname": "host3"},
            time=new_index,
        )
        t3[:] = np.array([i for i in range(13)]).reshape(13, 1)
        larger_t = t1.merge([t2, t3])

        merged_t = new_t.merge([larger_t])

        self.assertEqual(merged_t.shape, (13, 6))
        self.assertEqual(merged_t[12, 4], 12.0)
        self.assertTrue(np.isnan(merged_t[12, 1]))

        # create a smaller time series, starts before
        new_index = self.timeindex[2:8]
        t1 = TimeSeries(
            shape=(6, 1),
            labels={"name": "metric2", "env": "prod", "hostname": "host1"},
            time=new_index,
        )
        t1[:] = np.array([i for i in range(6)]).reshape(6, 1)
        t2 = TimeSeries(
            shape=(6, 1),
            labels={"name": "metric2", "env": "prod", "hostname": "host2"},
            time=new_index,
        )
        t2[:] = np.array([i for i in range(6)]).reshape(6, 1)
        t3 = TimeSeries(
            shape=(6, 1),
            labels={"name": "metric2", "env": "prod", "hostname": "host3"},
            time=new_index,
        )
        t3[:] = np.array([i for i in range(6)]).reshape(6, 1)
        smaller_t = t1.merge([t2, t3])

        merged_t = new_t.merge([smaller_t])

        self.assertEqual(merged_t.shape, (10, 6))
        self.assertEqual(merged_t[5, 1], 5.0)
        self.assertTrue(np.isnan(merged_t[9, 4]))

    def test_merge_time_index(self):
        t1, _, _ = self.basic_timeseries
        new_index = np.hstack(
            [
                self.timeindex,
                [x for x in range(int(self.timeindex[-1]) + 60, int(self.timeindex[-1]) + 240, 60)],
            ]
        )

        rtn1 = t1._merge_time_indicies(self.timeindex, new_index)
        self.assertTrue(np.array_equal(rtn1, new_index))
        self.assertTrue(
            np.array_equal(
                t1._merge_time_indicies(self.timeindex[[0, 1, 3, 5, 7]], new_index), new_index
            )
        )

    def test_merge_success(self):
        t1, t2, t3 = self.basic_timeseries
        values = [t1, t2, t3, t2, t2, t3]
        new_t = t1.merge(values[1:])
        for i, t in enumerate(values):
            self.assertTrue(np.array_equal(new_t[:, [i]], t))
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge(t2)
        self.assertTrue(np.array_equal(t2[:], new_t[:,1]))

        self.assertRaises(NotImplementedError, t2.merge, [0,1,2,3,4])


    def test_group_reduce(self):
        t1, t2, t3 = self.basic_timeseries
        t2 *= 3
        t3 *= 5

        # one label
        new_t = t1.merge([t2, t3, t2, t2, t3])
        reduced = new_t.group("hostname").add()
        host2 = new_t[:, [1]] + new_t[:, [3]] + new_t[:, [4]]
        self.assertTrue(np.array_equal(reduced[:, [1]], host2))

        # multiple labels
        new_t.labels[1]["name"] = "metric2"
        reduced = new_t.group(["hostname", "name"]).add()
        self.assertEqual(reduced.shape[1], 4)
        host2metric1 = new_t[:, [3]] + new_t[:, [4]]
        self.assertTrue(np.array_equal(reduced[:, [0]], new_t[:, [0]]))
        self.assertTrue(np.array_equal(reduced[:, [3]], host2metric1))

    def test_group_reduce_non_ufunc(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])
        reduced = new_t.group("env").mean()
        self.assertTrue(np.array_equal(reduced[:, [0]], avg_2))

    def test_ufunc_with_multiple_passes(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])
        reduced = new_t.group("env").mean()
        self.assertTrue(np.array_equal(reduced[:, [0]], avg_2))
        self.assertEqual(new_t.group("env").mean(axis=None), 13.5)

    def test_index_by_label(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        self.assertTrue(np.array_equal(new_t[{"hostname": "host2"}], new_t[:, [1]]))
        self.assertTrue(
            np.array_equal(new_t[{"hostname": ["host2", "host3"]}], new_t[:, [1, 2]])
        )
        self.assertTrue(
            np.array_equal(new_t[{"hostname": "does_not_exist"}], TimeSeries(shape=(0, 3)).data)
        )
        self.assertTrue(
            np.array_equal(new_t.filter({"hostname": "does_not_exist"}), TimeSeries(shape=(0, 3)).data)
        )

    def test_index_by_datetime(self):

        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        beg = np.datetime64(
            int(self.timeindex[2]), "s"
        )  # pylint: disable-msg=too-many-function-args
        end = np.datetime64(
            int(self.timeindex[5]), "s"
        )  # pylint: disable-msg=too-many-function-args

        # datetime range
        self.assertTrue(np.array_equal(new_t[beg:end], new_t[2:5]))

        # specific time
        self.assertTrue(np.array_equal(new_t[beg], new_t[2]))

        # set start time to the end
        self.assertTrue(np.array_equal(new_t[beg:], new_t[2:]))

        # beginning to set end time
        self.assertTrue(np.array_equal(new_t[:end], new_t[:5]))

        # get 3 minutes of data from start point
        self.assertTrue(
            np.array_equal(new_t[beg : np.timedelta64(3, "m")], new_t[2:5])
        )  # pylint: disable-msg=too-many-function-args

        # offset 3 minutes of data
        self.assertTrue(
            np.array_equal(new_t[np.timedelta64(3, "m") :], new_t[3:])
        )  # pylint: disable-msg=too-many-function-args

        # end 3 minutes before ending
        self.assertTrue(
            np.array_equal(new_t[: np.timedelta64(3, "m")], new_t[:6])
        )  # pylint: disable-msg=too-many-function-args

        # row by date, columns by number
        self.assertTrue(np.array_equal(new_t[beg:end, 1:], new_t[2:5, 1:]))

        # row by date, columns by number
        self.assertTrue(np.array_equal(new_t[beg, 1:], new_t[2, 1:]))

        # row by date, columns by number update
        new_t[beg:beg+1] = [0,0,0]
        self.assertTrue(np.array_equal([[0,0,0]], new_t[2]))

        new_t[beg] = [0,0,0]
        self.assertTrue(np.array_equal([[0,0,0]], new_t[2]))

    def test_panda_conversion(self):
        t1, t2, t3 = self.basic_timeseries
        t2 *= 3
        new_t = t1.merge([t2, t3])
        new_t.labels[2] = {
            "name": "metric1",
            "env": "prod",
            "hostname": "host3",
            "extratag": "tagger",
        }

        data, labels = new_t.to_pandas()

        # labels
        self.assertEqual(
            sorted(list(labels.columns)), sorted(["name", "env", "hostname", "extratag"])
        )
        self.assertEqual(list(labels["extratag"]), [None, None, "tagger"])

        # data
        self.assertTrue(np.array_equal(data.iloc[4,1], new_t[4, 1]))

    def test_weighted_rolling_window(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])

        rolling = new_t.rolling_window(3).average(weights=[1,2,3])
        answer = np.array([0.0, 0.0, 4./3., 7./3, 10./3, 13./3, 16./3, 19./3, 22./3, 25./3])
        answer2 = np.array([0.0, 0.0, 4, 7, 10, 13, 16, 19, 22, 25])
        answer3 = np.array([0.0, 0.0, 100./15., 175./15., 250./15., 325./15., 400./15., 475./15., 550./15., 625./15.])

        self.assertTrue(np.array_equal(rolling[2:, 0], answer[2:]))
        self.assertTrue(np.array_equal(rolling[2:, 1], answer2[2:]))
        self.assertTrue(np.array_equal(rolling[2:, 2], answer3[2:]))
       
    def test_rolling_window(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])

        rolling = new_t.rolling_window(3).median()
        answer = np.array([0.0, 0.0, 1, 2, 3, 4, 5, 6, 7, 8])
        answer2 = np.array([0.0, 0.0, 3, 6, 9, 12, 15, 18, 21, 24])
        answer3 = np.array([0.0, 0.0, 5, 10, 15, 20, 25, 30, 35, 40])

        self.assertTrue(np.array_equal(rolling[2:, 0], answer[2:]))
        self.assertTrue(np.array_equal(rolling[2:, 1], answer2[2:]))
        self.assertTrue(np.array_equal(rolling[2:, 2], answer3[2:]))

        rolling_mean = new_t.rolling_window(3).mean()
        self.assertTrue(np.array_equal(rolling[2:], rolling_mean[2:]))

        rolling_cum_sum = new_t.rolling_window(3).add()
        cum_sum_answer = np.array([0.0, 0.0, 3, 6, 9, 12, 15, 18, 21, 24])
        self.assertTrue(np.array_equal(rolling_cum_sum[2:,0], cum_sum_answer[2:]))
     
    def test_resample(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])

        resampled = new_t.resample(300).mean()
        resampled2 = new_t.resample('5m').mean()
        self.assertEqual(len(resampled),2)
        answer = [
            [2., 6., 10.],
            [7., 105./5., 175./5.]
        ]
        #0,1,2,3,4 - 5,6,7,8,9
        #0,3,6,9,12 - 15,18,21,24,27
        #0,5,10,15,20 - 25,30,35,40,45
        self.assertTrue(np.array_equal(resampled, answer))
        self.assertTrue(np.array_equal(resampled2, answer))

    def test_label_reduction(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        r = new_t.group('env').mean()
        self.assertEqual(r.labels, [{'env': 'prod', 'name': 'metric1'}])

        new_t.labels[1]['name'] = 'metric2'
        r = new_t.group('env').mean()
        self.assertEqual(r.labels, [{'env': 'prod'}])

    def test_label_reduction_2_groups(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        new_t.labels[1]['name'] = 'metric2'
        r = new_t.group('name').mean()
        self.assertEqual(r.labels, [{'name': 'metric1', 'env': 'prod'}, {'env': 'prod', 'hostname':'host2','name': 'metric2'}])

    def test_merge_labels_ufunc(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        a = np.add.reduce(new_t, axis=1)
        self.assertEqual(a.labels, [{'env': 'prod','name': 'metric1'}])
        new_t.labels[1]['name'] = 'metric2'
        a = np.add.reduce(new_t, axis=1)
        self.assertEqual(a.labels, [{'env': 'prod'}])



    def test_multiply_methods(self):
        t1, t2, t3 = self.basic_timeseries

        mul_result =  np.array([i * 3 for i in range(10)]).reshape(10, 1)

        ts = t1 * 3
        self.assertTrue(np.array_equal(ts.data, mul_result))
        ts = 3 * t2
        self.assertTrue(np.array_equal(ts.data, mul_result))
        t3 *= 3
        self.assertTrue(np.array_equal(t3.data, mul_result))

    def test_ufunc_attached(self):
        t1, t2, t3 = self.basic_timeseries
        t1 *= -1
        new_t = t1.fabs()
        self.assertTrue(np.array_equal(t2.data, new_t.data))

    def test_invalid_attribute(self):
        t1, t2, t3 = self.basic_timeseries
        self.assertRaises(AttributeError, t1.invalid_name_xxx)

    def test_invalid_index(self):
        t1, t2, t3 = self.basic_timeseries

        # out of bounds index
        self.assertRaises(IndexError, t1.__getitem__, 2300)

        # out of bounds date index
        out_of_bounds_date = np.datetime64(int(t1.time[-1] + 86400), "s")
        self.assertRaises(IndexError, t1.__getitem__, out_of_bounds_date)

        # out of bounds stop date index
        out_of_bounds_date = np.datetime64(int(t1.time[0] - 86400), "s")
        new_t = t1[out_of_bounds_date:np.timedelta64(3, "m")]
        self.assertEqual(len(new_t), 0)

    def test_matplotlib(self):
        t1, t2, t3 = self.basic_timeseries

        t1.plot(legend=True)
        self.assertEqual(1, len(plt.get_fignums()))
