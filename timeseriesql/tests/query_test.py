import unittest
import numpy as np
from timeseriesql.query import Query, Plan


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_gather_period(self):
        t1 = Query(x.mean for x in "test")[3600]
        t1_timeindex = np.array(
            [np.float64(t1.time[0] + x) for x in range(0, 3600, Query().DEFAULT_RESOLUTION)],
            dtype=np.float64,
        )
        t2 = Query(x.mean for x in "test")[3600:1800]
        t2_timeindex = np.array(
            [np.float64(t2.time[0] + x) for x in range(0, 1800, Query().DEFAULT_RESOLUTION)],
            dtype=np.float64,
        )
        t3 = Query(x.mean for x in "test")[3600:1800:300]
        t3_timeindex = np.array(
            [np.float64(t3.time[0] + x) for x in range(0, 1800, 300)], dtype=np.float64
        )
        t4 = Query(x.mean for x in "test").fetch()
        t4_timeindex = np.array(
            [np.float64(t4.time[0] + x) for x in range(0, 3600, Query().DEFAULT_RESOLUTION)],
            dtype=np.float64,
        )
        t5 = Query(x.mean for x in "test")["1h"]

        self.assertTrue(np.array_equal(t1.time, t1_timeindex))
        self.assertTrue(np.array_equal(t2.time, t2_timeindex))
        self.assertTrue(np.array_equal(t3.time, t3_timeindex))
        self.assertTrue(np.array_equal(t4.time, t4_timeindex))
        self.assertTrue(np.array_equal(t5.time, t1_timeindex))
