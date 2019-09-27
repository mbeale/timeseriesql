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

        t1 = TimeSeries((10, 2), labels={"name": "metric1", "env": "prod", "hostname": "host1"})
        t1[:] = [[i, v] for i, v in zip(timeindex, range(10))]
        t2 = TimeSeries((10, 2), labels={"name": "metric1", "env": "prod", "hostname": "host2"})
        t2[:] = [[i, v] for i, v in zip(timeindex, range(10))]
        t3 = TimeSeries((10, 2), labels={"name": "metric1", "env": "prod", "hostname": "host3"})
        t3[:] = [[i, v] for i, v in zip(timeindex, range(10))]

        self.timeindex = timeindex
        self.basic_timeseries = (t1, t2, t3)

    def test_init_array(self):
        shape = (8, 9)
        t = TimeSeries(shape)
        self.assertEqual(shape, t.shape)

    def test_add_stream_full(self):

        t1, t2, t3 = self.basic_timeseries

        new_t = t1.merge([t2, t3])
        self.assertTrue(np.array_equal(self.timeindex, new_t.time))
        self.assertEqual(new_t[2, 1], 2.0)
        self.assertEqual(new_t[2, 2], 2.0)
        self.assertEqual(new_t[2, 3], 2.0)

    def test_reduce_add_internal(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        reduced_t = np.add.reduce(new_t, 1)

        self.assertEqual(reduced_t[2, 1], 6)
        self.assertEqual(reduced_t[3, 1], 9)
        self.assertEqual(reduced_t[4, 1], 12)

    def test_add_scalar(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        rtn_new = new_t + 5
        self.assertEqual(rtn_new[0, 0], t2[0, 0])
        self.assertEqual(rtn_new[0, 1], 5.0)
        self.assertEqual(rtn_new[1, 2], 6.0)
        self.assertEqual(rtn_new[2, 3], 7.0)
        self.assertEqual(rtn_new[3, 2], 8.0)

        new_t += 5

        self.assertEqual(new_t[0, 1], 5.0)
        self.assertEqual(new_t[1, 2], 6.0)
        self.assertEqual(new_t[2, 3], 7.0)
        self.assertEqual(new_t[3, 2], 8.0)

    def test_combine_two_series_with_add(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        new_t += copy.deepcopy(new_t)

        self.assertEqual(new_t.shape, (10, 4))
        self.assertEqual(new_t[3, 2], 6)
        self.assertEqual(new_t[4, 1], 8)

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

        t1 = TimeSeries((13, 2), labels={"name": "metric2", "env": "prod", "hostname": "host1"})
        t1[:] = [[i, v] for i, v in zip(new_index, range(13))]
        t2 = TimeSeries((13, 2), labels={"name": "metric2", "env": "prod", "hostname": "host2"})
        t2[:] = [[i, v] for i, v in zip(new_index, range(13))]
        t3 = TimeSeries((13, 2), labels={"name": "metric2", "env": "prod", "hostname": "host3"})
        t3[:] = [[i, v] for i, v in zip(new_index, range(13))]
        larger_t = t1.merge([t2, t3])

        merged_t = new_t.merge([larger_t])

        self.assertEqual(merged_t.shape, (13, 7))
        self.assertEqual(merged_t[12, 5], 12.0)
        self.assertTrue(np.isnan(merged_t[12, 2]))

        # create a smaller time series, starts before
        new_index = self.timeindex[2:8]
        t1 = TimeSeries((6, 2), labels={"name": "metric2", "env": "prod", "hostname": "host1"})
        t1[:] = [[i, v] for i, v in zip(new_index, range(6))]
        t2 = TimeSeries((6, 2), labels={"name": "metric2", "env": "prod", "hostname": "host2"})
        t2[:] = [[i, v] for i, v in zip(new_index, range(6))]
        t3 = TimeSeries((6, 2), labels={"name": "metric2", "env": "prod", "hostname": "host3"})
        t3[:] = [[i, v] for i, v in zip(new_index, range(6))]
        smaller_t = t1.merge([t2, t3])

        merged_t = new_t.merge([smaller_t])

        self.assertEqual(merged_t.shape, (10, 7))
        self.assertEqual(merged_t[5, 2], 5.0)
        self.assertTrue(np.isnan(merged_t[9, 5]))

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
            self.assertTrue(np.array_equal(new_t[:, [i + 1]], t.get_data))

    def test_group_reduce(self):
        t1, t2, t3 = self.basic_timeseries
        t2 *= 3
        t3 *= 5

        # one label
        new_t = t1.merge([t2, t3, t2, t2, t3])
        reduced = new_t.group("hostname")
        host2 = new_t[:, [0, 2]] + new_t[:, [0, 4]] + new_t[:, [0, 5]]
        self.assertTrue(np.array_equal(reduced[:, [0, 2]], host2))

        # multiple labels
        new_t.labels[1]["name"] = "metric2"
        reduced = new_t.group(["hostname", "name"])
        self.assertEqual(reduced.shape[1], 5)
        host2 = new_t[:, [0, 4]] + new_t[:, [0, 5]]
        self.assertTrue(np.array_equal(reduced[:, [0, 2]], new_t[:, [0, 2]]))
        self.assertTrue(np.array_equal(reduced[:, [0, 4]], host2))


        #reduced_collect = new_t.group_collect(["hostname", "name"]).add()
        #self.assertTrue(np.array_equal(reduced, reduced_collect))

    def test_group_reduce_non_ufunc(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])
        reduced = new_t.group("env", lambda x: np.add.reduce(x, axis=1) / (x.shape[1]))
        self.assertTrue(np.array_equal(reduced[:, [0, 1]], avg_2))

    def test_ufunc_with_multiple_passes(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])
        reduced = new_t.group("env", lambda x: np.mean(x, axis=1))
        self.assertTrue(np.array_equal(reduced[:, [0, 1]], avg_2))
        self.assertEqual(new_t.group("env", np.mean), 13.5)

    def test_index_by_label(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        self.assertTrue(np.array_equal(new_t[{"hostname": "host2"}], new_t[:, [0, 2]]))
        self.assertTrue(
            np.array_equal(new_t[{"hostname": ["host2", "host3"]}], new_t[:, [0, 2, 3]])
        )
        self.assertIsNone(new_t[{"hostname": "does_not_exist"}])

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
        new_t[beg:beg+1, 1:] = [0,0,0]
        self.assertTrue(np.array_equal([0,0,0], new_t[2, 1:]))

    def test_panda_conversion(self):
        t1, t2, t3 = self.basic_timeseries
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
        self.assertTrue(np.array_equal(data["stream_0"], new_t[:, 1]))

    def test_rolling_window(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])

        rolling = new_t.rolling_window(3, lambda x: np.median(x, axis=0))
        answer = np.array([0.0, 0.0, 1, 2, 3, 4, 5, 6, 7, 8])
        answer2 = np.array([0.0, 0.0, 3, 6, 9, 12, 15, 18, 21, 24])
        answer3 = np.array([0.0, 0.0, 5, 10, 15, 20, 25, 30, 35, 40])

        self.assertTrue(np.array_equal(rolling[2:, 1], answer[2:]))
        self.assertTrue(np.array_equal(rolling[2:, 2], answer2[2:]))
        self.assertTrue(np.array_equal(rolling[2:, 3], answer3[2:]))

        rolling_mean = new_t.rolling_window(3, lambda x: np.mean(x, axis=0))
        self.assertTrue(np.array_equal(rolling[2:], rolling_mean[2:]))

        rolling_cum_sum = new_t.rolling_window(3, np.add.reduce)
        cum_sum_answer = np.array([0.0, 0.0, 3, 6, 9, 12, 15, 18, 21, 24])
        self.assertTrue(np.array_equal(rolling_cum_sum[2:,1], cum_sum_answer[2:]))
     
    def test_resample(self):
        avg_1, avg_2, avg_3 = self.basic_timeseries
        avg_2 *= 3
        avg_3 *= 5
        new_t = avg_1.merge([avg_2, avg_3])

        resampled = new_t.resample(300)
        self.assertEqual(len(resampled),2)
        answer = [
            [2., 6., 10.],
            [7., 105./5., 175./5.]
        ]
        #0,1,2,3,4 - 5,6,7,8,9
        #0,3,6,9,12 - 15,18,21,24,27
        #0,5,10,15,20 - 25,30,35,40,45
        self.assertTrue(np.array_equal(resampled[:,1:], answer))

    def test_label_reduction(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        r = new_t.group('env')
        self.assertEqual(r.labels, [{'env': 'prod', 'name': 'metric1'}])

        new_t.labels[1]['name'] = 'metric2'
        r = new_t.group('env')
        self.assertEqual(r.labels, [{'env': 'prod'}])

    def test_label_reduction_2_groups(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])
        new_t.labels[1]['name'] = 'metric2'
        r = new_t.group('name')
        self.assertEqual(r.labels, [{'name': 'metric1', 'env': 'prod'}, {'env': 'prod', 'hostname':'host2','name': 'metric2'}])

    def test_merge_labels_ufunc(self):
        t1, t2, t3 = self.basic_timeseries
        new_t = t1.merge([t2, t3])

        a = np.add.reduce(new_t, axis=1)
        self.assertEqual(a.labels, [{'env': 'prod','name': 'metric1'}])
        new_t.labels[1]['name'] = 'metric2'
        a = np.add.reduce(new_t, axis=1)
        self.assertEqual(a.labels, [{'env': 'prod'}])




       
