import unittest
import numpy as np
from timeseriesql.query import Query, Plan


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "__slots__"):
        return todict({s: getattr(obj, s, None) for s in obj.__slots__})
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey)) 
            for key, value in obj.__dict__.items() 
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

class TestQuery(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def test_basic_query(self):
        q = Query(x for x in "test")
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_basic_query_with_label(self):
        q = Query(x.mean for x in "test")
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [],
                "group": None,
                "variables": [{'name': 'x', 'labels': ['mean']}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_basic_filter(self):
        self.maxDiff = None

        q = Query(x for x in "test" if x.label1 == 'prod')
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': 'prod'}}],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_binary_operations(self):
        q = Query(x * 100 for x in "test" if x.label1 == 'prod')
        expected_plan = {
                "metrics": ["test"],
                "calc": [['x', 100, 'BINARY_MULTIPLY']],
                "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': 'prod'}}],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_group_by(self):
        q = Query(x * 100 for x in "test" if x.label1 == 'prod').by('label1')
        expected_plan = {
                "metrics": ["test"],
                "calc": [['x', 100, 'BINARY_MULTIPLY']],
                "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': 'prod'}}],
                "group": [['label1'], 'mean'],
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_multiple_binary_operations(self):
        q = Query(x * 100 * x for x in "test" if x.label1 == 'prod')
        expected_plan = {
                "metrics": ["test"],
                "calc": [['x', 100, 'BINARY_MULTIPLY'], ['x', 'BINARY_MULTIPLY']],
                "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': 'prod'}}],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)


    def test_multiple_generators(self):
        self.maxDiff = None
        q = Query(x - y for x,y in Query((x.max for x in "metric1"), (x.min for x in "metric2")).by('tag1'))
        expected_plan = {
                "metrics": [
                    {
                        "calc": None,
                        "metrics": ["metric1"],
                        "filters": [],
                        "group": [['tag1'], 'mean'],
                        "variables": [{'name': 'x', 'labels': ['max']}]
                    },{
                        "calc": None,
                        "metrics": ["metric2"],
                        "filters": [],
                        "group": [['tag1'], 'mean'],
                        "variables": [{'name': 'x', 'labels': ['min']}]
                    }
                ],
                "calc": [['x', 'y', 'BINARY_SUBTRACT']],
                "filters": [],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}, {'name': 'y', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_gather_period(self):
        t1 = Query(x.mean for x in "test")[3600]
        t1_timeindex = np.array([np.float64(t1[0,0] + x) for x in range(0, 3600, Query().DEFAULT_RESOLUTION)], dtype=np.float64)
        t2 = Query(x.mean for x in "test")[3600:1800]
        t2_timeindex = np.array([np.float64(t2[0,0] + x) for x in range(0, 1800, Query().DEFAULT_RESOLUTION)], dtype=np.float64)
        t3 = Query(x.mean for x in "test")[3600:1800:300]
        t3_timeindex = np.array([np.float64(t3[0,0] + x) for x in range(0, 1800, 300)], dtype=np.float64)
        t4 = Query(x.mean for x in "test").fetch()
        t4_timeindex = np.array([np.float64(t4[0,0] + x) for x in range(0, 3600, Query().DEFAULT_RESOLUTION)], dtype=np.float64)

        self.assertTrue(np.array_equal(t1.time, t1_timeindex))   
        self.assertTrue(np.array_equal(t2.time, t2_timeindex))   
        self.assertTrue(np.array_equal(t3.time, t3_timeindex))   
        self.assertTrue(np.array_equal(t4.time, t4_timeindex))   

    def test_query_with_varname_in_filter(self):
        value = 'prod'
        q = Query(x for x in "test" if x.label1 == value)
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': 'prod'}}],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_query_with_string_variable_for_metric(self):
        value = 'test'
        q = Query(x for x in value if x.label1 == 'prod')
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': 'prod'}}],
                "group": None,
                "variables": [{'name': 'x', 'labels': []}]
        }
        self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

    def test_for_loop_filter_variables(self):
        filters = ['filter1', 'filter2','filter3']
        for i in range(len(filters)):
            q = Query(x for x in "test" if x.label1 == filters[i])
            expected_plan = {
                        "metrics": ["test"],
                        "calc": None,
                        "filters": [{'left': {'type': 'var', 'name': 'x', 'labels': ['label1']}, 'op': '==', 'right': {'type': 'string', 'value': filters[i]}}],
                        "group": None,
                        "variables": [{'name': 'x', 'labels': []}]
                }
            self.assertEqual(todict(q._generate_plan()[0]), expected_plan)

