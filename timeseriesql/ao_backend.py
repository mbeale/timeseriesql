import os
import requests
import numbers
from .query import Query, Plan
from .timeseries import TimeSeries


def create_scalar_time_series(series, scalar):
    """
    use a trick to create a scalar time series by dividing a timeseries by itself and then 
    scaling it by the desired scalar
    """
    return f"scale(divide([{series},{series}]),{{\"factor\":\"{scalar}\"}})"

def binary_operation(left, right, optype):
    """ Handle binary operations """
    op_handlers = {
        'BINARY_MULTIPLY': 'multiply',
        'BINARY_SUBTRACT': 'subtract',
        'BINARY_POWER': None,
        'BINARY_TRUE_DIVIDE': 'divide',
        'BINARY_FLOOR_DIVIDE': 'divide',
        'BINARY_MATRIX_MULTIPLY': 'multiply',
        'BINARY_MODULO':None,
        'BINARY_ADD':'sum',
        'BINARY_SUBSCR': None,
        'BINARY_LSHIFT': None,
        'BINARY_RSHIFT': None,
        'BINARY_AND': None,
        'BINARY_XOR': None,
        'BINARY_OR': None
    }

    is_right_scalar = isinstance(right, numbers.Number)
    is_left_scalar = isinstance(left, numbers.Number)

    opcode = op_handlers.get(optype, None)
    if opcode:
        if is_left_scalar:
            return CompositeDefinition(f"{opcode}([{create_scalar_time_series(right, left)}, {right}])")
        elif is_right_scalar:
            return CompositeDefinition(f"{opcode}([{left},{create_scalar_time_series(left, right)}])")
        else:
            return CompositeDefinition(f"{opcode}([{left},{right}])")
    else:
        raise TypeError("{optype} is not a supported operation")


class CompositeDefinition(str):
    """A class for composite definitions"""
    pass        

class AOBackend(Query):
    """ Extends the Query object to translate execution plans into the AppOptics Composite language """
    BASE_URL = "https://api.appoptics.com/v1/"
    API_TOKEN = os.environ.get('APPOPTICS_TOKEN', '')

    @classmethod
    def get(cls, endpoint, params):
        """issue a get request"""
        query_str = "?" + "&".join(["{}={}".format(key, value) for key, value in params.items()])
        response = requests.get(
            cls.BASE_URL+endpoint+query_str,
            auth=requests.auth.HTTPBasicAuth(
                cls.API_TOKEN,
                ''))
        if response.status_code > 399:
            raise ConnectionError("{} - {}".format(response.status_code, response.text))
        return response

    @classmethod
    def post(cls, endpoint, body):
        """issue a post request"""
        response = requests.post(
            cls.BASE_URL+endpoint,
            json=body,
            auth=requests.auth.HTTPBasicAuth(
                cls.API_TOKEN,
                ''))
        if response.status_code > 399:
            raise ValueError("{} - {}".format(response.status_code, response.text))
        return response

    def create_query(self, plan, period):
        """Create a composite based on the plan and period"""
        query = ""
        values = []
        for i, m in enumerate(plan.metrics):
            if isinstance(m, Plan):
                values.append(AOBackend().create_query(m, period))
            else:
                label_size = len(plan.variables[i]['labels'])
                if  label_size > 1:
                    raise AttributeError('Only one label is supported for stream featching')
                series_func = "mean" if label_size == 0 else plan.variables[i]['labels'][0]
                series_filter = "\"*\""
                if len(plan.filters) > 0:
                    series_filter = '{'
                    for f in plan.filters:
                        if f['op'] != '==':
                            raise AttributeError(f"Only BINARY_EQUAL options are supported, {f['op']} received")
                        var_loc = [k for k in f.keys() if k != 'op' if f[k]['type'] == 'var']
                        string_loc = [k for k in f.keys() if k != 'op' if f[k]['type'] == 'string']
                        if len(var_loc) == 0 or len(string_loc) == 0:
                            raise AttributeError(f"Must have one string and one label allowed in the filter")
                        series_filter += f"\"{f[var_loc[0]]['labels'][0]}\":\"{f[string_loc[0]]['value']}\""
                    series_filter += '}'
                series_def = CompositeDefinition(f"s(\"{m}\",{series_filter},{{period:\"{period['resolution']}\",\"function\":\"{series_func}\"}})")
                if plan.group:
                    aggr_func = plan.group[1]
                    series_def = CompositeDefinition(f"{aggr_func}({series_def})")
                values.append(series_def)
        value_len = len(values)
        if value_len == 1 and plan.calc is None:
            query = values[0]
        elif value_len == 0:
            raise AttributeError("A valid query could not be created for this expression")
        elif plan.calc is None:
            query = CompositeDefinition(f"[{','.join([v for v in values])}]")
        else:
            for e in plan.calc:
                if len(e) == 3:
                    left = e[0] if isinstance(e[0], numbers.Number) else [values[i] for i,n in enumerate(plan.variables)if n['name'] == e[0]][0]
                    right = e[1] if isinstance(e[1], numbers.Number) else [values[i] for i,n in enumerate(plan.variables)if n['name'] == e[1]][0]
                    query = binary_operation(left, right, e[2])
                else:
                    right = e[0] if isinstance(e[0], numbers.Number) else [values[i] for i,n in enumerate(plan.variables)if n['name'] == e[0]][0]
                    query = binary_operation(query, right, e[1])
        if plan.group:
            labels = ','.join([f"\"{l}\"" for l in plan.group[0]])
            query = CompositeDefinition(f"group_by({labels},{query})")
        print(query)
        return query

    def execute_plan(self):
        """Execute the plan and return a TimeSeries object for any further processing or displaying"""
        params = self._process_period()
        plan = self._generate_plan()[0]
        params['compose'] = self.create_query(plan, params)
        series = self.get('measurements', params)
        timeseries = None
        for stream in series.json()['series']:
            labels = {**stream['tags']}
            if 'metric' in stream:
                labels['name'] = stream['metric']['name']
            t = TimeSeries((len(stream['measurements']),2), labels=labels)
            t[:] = [[m['time'],m['value']] for m in stream['measurements']]
            if timeseries is not None:
                timeseries = timeseries.merge([t])
            else:
                timeseries = t.copy()
        return timeseries


