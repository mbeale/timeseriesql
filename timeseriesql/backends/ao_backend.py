import os
import requests
import numbers
import time
import re
from timeseriesql.query import Query
from timeseriesql.timeseries import TimeSeries
from timeseriesql.ast import Metric, Value


def create_scalar_time_series(series, scalar):
    """
    use a trick to create a scalar time series by dividing a timeseries by itself and then 
    scaling it by the desired scalar
    """
    return f'scale(divide([{series},{series}]),{{"factor":"{scalar}"}})'


def power_time_series(series, scalar):
    """ Multiply a series by itself X times where X is a scalar """
    s = str(series)
    return f"multiply([{','.join([s for _ in range(scalar)])}])"

def modulo_time_series(series, scalar):
    """ Get the modulo of series.  num - divisor * floor(num / divisor)) """
    scalar_series = create_scalar_time_series(series,scalar)
    return f"subtract([{series},multiply([{scalar_series},floor(divide([{series},{scalar_series}]))])])"

def binary_operation(left, right, optype):
    """ Handle binary operations """
    op_handlers = {
        "BINARY_MULTIPLY": "multiply",
        "BINARY_SUBTRACT": "subtract",
        "BINARY_POWER": "power",
        "BINARY_TRUE_DIVIDE": "divide",
        "BINARY_FLOOR_DIVIDE": "divide",
        "BINARY_MATRIX_MULTIPLY": "multiply",
        "BINARY_MODULO": "modulo",
        "BINARY_ADD": "sum",
    }

    is_right_scalar = isinstance(right, numbers.Number)
    is_left_scalar = isinstance(left, numbers.Number)

    opcode = op_handlers.get(optype, None)
    if opcode:
        if opcode in ["power", "modulo"]:
            op_func = power_time_series
            if opcode == 'modulo':
                op_func = modulo_time_series
            if is_left_scalar:
                return op_func(right, left)
            elif is_right_scalar:
                return op_func(left, right)
            else:
                return ValueError(f"There was not a scalar present for the {opcode} operation")
        else:
            if is_left_scalar:
                return f"{opcode}([{create_scalar_time_series(right, left)}, {right}])"
            elif is_right_scalar:
                return f"{opcode}([{left},{create_scalar_time_series(left, right)}])"
            else:
                return f"{opcode}([{left},{right}])"
    raise TypeError(f"{optype} is not a supported operation")


class CompositeDefinition:
    """A class for composite definitions"""

    def __init__(self, name, resolution=1):
        self.name = name
        self.filter = []
        self.resolution = resolution
        self.sum_func = "mean"

    def __str__(self):
        filters = '"*"'
        if len(self.filter) > 0:
            filters = "{"
            quote = '"'
            for x in self.filter:
                val = x["value"] if x["op"] == "==" else "!" + x["value"]
                filters += f"{quote}{x['name']}{quote}:{quote}{val}{quote}"
            filters += "}"
        return f's("{self.name}",{filters},{{period:"{self.resolution}","function":"{self.sum_func}"}})'


class AOBackend(Query):
    """ Extends the Query object to translate execution plans into the AppOptics Composite language """

    BASE_URL = "https://api.appoptics.com/v1/"
    API_TOKEN = os.environ.get("APPOPTICS_TOKEN", "")
    COMPOSITE_DEF = None

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            self.COMPOSITE_DEF = args[0]
        super(AOBackend, self).__init__(*args)

    @classmethod
    def get(cls, endpoint, params):
        """issue a get request"""
        query_str = "?" + "&".join(["{}={}".format(key, value) for key, value in params.items()])
        response = requests.get(
            cls.BASE_URL + endpoint + query_str, auth=requests.auth.HTTPBasicAuth(cls.API_TOKEN, "")
        )
        if response.status_code > 399:
            raise ConnectionError("{} - {}".format(response.status_code, response.text))
        return response

    @classmethod
    def post(cls, endpoint, body):
        """issue a post request"""
        response = requests.post(
            cls.BASE_URL + endpoint, json=body, auth=requests.auth.HTTPBasicAuth(cls.API_TOKEN, "")
        )
        if response.status_code > 399:
            raise ValueError("{} - {}".format(response.status_code, response.text))
        return response

    @property
    def composite(self):
        if not self.COMPOSITE_DEF:
            now = int(time.time())
            plan = self._generate_plan()
            self.COMPOSITE_DEF = self.create_query(
                plan, {"start_time": now - 3600, "end_time": now, "resolution": 1}
            )
        return self.COMPOSITE_DEF

    #  AST Class handling

    def binaryadd(self, left, right, period):
        return binary_operation(left, right, "BINARY_ADD")

    def binaryfloordivide(self, left, right, period):
        return binary_operation(left, right, "BINARY_FLOOR_DIVIDE")

    def binarymatrixmultiply(self, left, right, period):
        return binary_operation(left, right, "BINARY_MATRIX_MULTIPLY")

    def binarymodulo(self, left, right, period):
        return binary_operation(left, right, "BINARY_MODULO")

    def binarymultiply(self, left, right, period):
        return binary_operation(left, right, "BINARY_MULTIPLY")

    def binarypower(self, left, right, period):
        return binary_operation(left, right, "BINARY_POWER")

    def binarysubtract(self, left, right, period):
        return binary_operation(left, right, "BINARY_SUBTRACT")

    def binarytruedivide(self, left, right, period):
        return binary_operation(left, right, "BINARY_TRUE_DIVIDE")

    def compareequal(self, left, right, period):
        return {"name": left, "value": right, "op": "=="}

    def comparenotequal(self, left, right, period):
        return {"name": left, "value": right, "op": "!="}

    def filter(self, left, right, period):
        left.filter.append(right)
        return left

    def funcargs(self, left, right, period):
        kwargs = ""
        if len(right) > 0:
            kwargs = ",{"
            for k, v in right.items():
                if kwargs != ",{":
                    kwargs += ","
                kwargs += '"' + k + '":"' + str(v) + '"'
            kwargs += "}"
        left = f"{self.traverse_tree(left[0], period)}"
        return f"{left}{kwargs}"

    def funccall(self, left, right, period):
        valid_func_names = [
            "abs",
            "bottom",
            "ceiling",
            "derive",
            "fill",
            "filter",
            "floor",
            "integrate",
            "last_fill",
            "max",
            "mean",
            "min",
            "moving_average",
            "rate",
            "sum",
            "top",
            "window",
            "zero_fill",
        ]
        if callable(left):
            left = left.__name__
        if left not in valid_func_names:
            raise NotImplementedError(f"{left} is not a valid function")
        return f"{left}({right})"

    def group(self, left, right, period):
        if len(right[0]) == 0:
            raise AttributeError("Must have at least one label to group by")
        labels = '"' + ",".join(right[0]) + '"'
        if isinstance(left, str):
            sets = []
            for m in re.finditer("s\(([^)]+)\)", left):
                sets.append((m.start(), m.end()))
            for s in reversed(sets):
                left = left[0 : s[0]] + f"mean({left[s[0]:s[1]]})" + left[s[1] :]
        return f"group_by({labels},{left})"

    def loadattr(self, left, right, period):
        if isinstance(left, CompositeDefinition):
            left.sum_func = right
            return left
        raise NotImplementedError

    # end AST class handling

    def traverse_tree(self, root, period):
        if root:
            if isinstance(root, Metric):
                return CompositeDefinition(root.value, resolution=period["resolution"])
            if isinstance(root, Value):
                return root.value
            else:
                left = self.traverse_tree(root.left, period)
                right = self.traverse_tree(root.right, period)
                try:
                    op = self.__getattribute__(root.__class__.__name__.lower())
                    return op(left, right, period)
                except Exception as e:
                    raise NotImplementedError(
                        f"AST class of {root.__class__.__name__} is not supported"
                    )

    def create_query(self, plan, period):
        """Create a composite based on the plan and period"""

        query = ""
        query = self.traverse_tree(plan, period)
        return str(query)

    def execute_plan(self):
        """Execute the plan and return a TimeSeries object for any further processing or displaying"""
        params = self._process_period()
        if self.COMPOSITE_DEF:
            params["compose"] = self.COMPOSITE_DEF
        else:
            plan = self._generate_plan()
            self.COMPOSITE_DEF = self.create_query(plan, params)
            params["compose"] = self.COMPOSITE_DEF
        series = self.get("measurements", params)
        timeseries = None
        for stream in series.json()["series"]:
            labels = {**stream["tags"]}
            if "metric" in stream:
                labels["name"] = stream["metric"]["name"]
            time, data = zip(*[(m["time"], m["value"]) for m in stream["measurements"]])
            t = TimeSeries(shape=(len(stream["measurements"]), 1), labels=labels, time=time)
            t[:, 0] = data
            if timeseries is not None:
                timeseries = timeseries.merge([t])
            else:
                timeseries = t.copy()
        return timeseries
