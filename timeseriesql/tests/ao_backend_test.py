import unittest
import time
from timeseriesql.ao_backend import AOBackend, create_scalar_time_series
from timeseriesql.query import Plan
from timeseriesql.decompiler import Variable, FilterVariable, Constant


class TestAOBackend(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        now = int(time.time())
        self.basic_period = {
            "start_time": now - 3600,
            "end_time": now,
            "resolution": 1
        }

    def test_execute_basic_plan(self):
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [],
                "group": None,
                "variables": [Variable('x', labels=[])]
        }

        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), 's("test","*",{period:"1","function":"mean"})')

    def test_basic_query_with_label(self):
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [],
                "group": None,
                "variables": [Variable('x', labels=['max'])]
        }
        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), 's("test","*",{period:"1","function":"max"})')

    def test_basic_filter(self):
        expected_plan = {
                "metrics": ["test"],
                "calc": None,
                "filters": [{'left': FilterVariable(Variable('x', labels=['label1'])), 'op': '==', 'right': FilterVariable(Constant('prod'))}],
                "group": None,
                "variables": [Variable('x', labels= [])]
        }
        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), 's("test",{"label1":"prod"},{period:"1","function":"mean"})')

    def test_binary_operations(self):
        expected_plan = {
                "metrics": ["test"],
                "calc": [['x', 100, 'BINARY_MULTIPLY']],
                "filters": [{'left': FilterVariable(Variable('x', labels=['label1'])), 'op': '==', 'right': FilterVariable(Constant('prod'))}],
                "group": None,
                "variables": [Variable('x', labels= [])]
        }
        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), 'multiply([s("test",{"label1":"prod"},{period:"1","function":"mean"}),scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{\"factor\":"100"})])')

    def test_group_by(self):
        expected_plan = {
                "metrics": ["test"],
                "calc": [['x', 100, 'BINARY_MULTIPLY']],
                "filters": [{'left': FilterVariable(Variable('x', labels=['label1'])), 'op': '==', 'right': FilterVariable(Constant('prod'))}],
                "group": [['label1'], 'mean'],
                "variables": [Variable('x', labels= [])]
        }
        expected_value = 'group_by("label1",multiply([mean(s("test",{"label1":"prod"},{period:"1","function":"mean"})),scale(divide([mean(s("test",{"label1":"prod"},{period:"1","function":"mean"})),mean(s("test",{"label1":"prod"},{period:"1","function":"mean"}))]),{"factor":"100"})]))'

        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), expected_value)

    def test_multiple_binary_operations(self):
        expected_plan = {
                "metrics": ["test"],
                "calc": [['x', 100, 'BINARY_MULTIPLY'], ['x', 'BINARY_MULTIPLY']],
                "filters": [{'left': FilterVariable(Variable('x', labels=['label1'])), 'op': '==', 'right': FilterVariable(Constant('prod'))}],
                "group": None,
                "variables": [Variable('x', labels= [])]
        }
        expected_value = 'multiply([multiply([s("test",{"label1":"prod"},{period:"1","function":"mean"}),scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"})]),s("test",{"label1":"prod"},{period:"1","function":"mean"})])'

        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), expected_value)

    def test_scaler_create_trick(self):
        self.maxDiff = None
        series = 's("AWS.EC2.CPUUtilization",{"hostname":"ip-172-26-26-192.ec2.internal"},{period:"60","function":"mean"})'
        expected_result = 'scale(divide([s("AWS.EC2.CPUUtilization",{"hostname":"ip-172-26-26-192.ec2.internal"},{period:"60","function":"mean"}),s("AWS.EC2.CPUUtilization",{"hostname":"ip-172-26-26-192.ec2.internal"},{period:"60","function":"mean"})]),{"factor":"100"})'

        actual_result = create_scalar_time_series(series, 100)
        self.assertEqual(actual_result, expected_result)

    def test_multiple_generators(self):
        self.maxDiff = None
        expected_plan = {
                "metrics": [
                    Plan(**{
                        "calc": None,
                        "metrics": ["metric1"],
                        "filters": [],
                        "group": [['tag1'], 'mean'],
                        "variables": [Variable('x', labels= ['max'])]
                    }),Plan(**{
                        "calc": None,
                        "metrics": ["metric2"],
                        "filters": [],
                        "group": [['tag1'], 'mean'],
                        "variables": [Variable('x', labels=['min'])]
                    })
                ],
                "calc": [['x', 'y', 'BINARY_SUBTRACT']],
                "filters": [],
                "group": None,
                "variables": [Variable('x', labels= []), Variable('y', labels= [])]
        }
        expected_value = 'subtract([group_by("tag1",mean(s("metric1","*",{period:"1","function":"max"}))),group_by("tag1",mean(s("metric2","*",{period:"1","function":"min"})))])'
        self.assertEqual(AOBackend().create_query(Plan(**expected_plan), self.basic_period), expected_value)


        