import unittest
import numpy as np
import time
from timeseriesql.query import Query


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

        now = int(time.time())
        t6 = Query(x.mean for x in "test").range(start=now - 3600, end=now - 1800, resolution=1)

        self.assertTrue(np.array_equal(t1.time, t1_timeindex))
        self.assertTrue(np.array_equal(t2.time, t2_timeindex))
        self.assertTrue(np.array_equal(t3.time, t3_timeindex))
        self.assertTrue(np.array_equal(t4.time, t4_timeindex))
        self.assertTrue(np.array_equal(t5.time, t1_timeindex))
        self.assertTrue(np.array_equal(t6.time, t2_timeindex))

        with self.assertRaises(IndexError):
            Query(x.mean for x in "test")["1hkbkgvk"]

        with self.assertRaises(IndexError):
            Query(x.mean for x in "test")[1.04884]

    def test_iteration(self):
        i = 0
        q = Query(x.mean for x in "test")
        for i, _ in enumerate(q):
            pass
        self.assertEqual(0, i)

if __name__ == '__main__':
    unittest.main()
