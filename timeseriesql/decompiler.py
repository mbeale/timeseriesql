import dis


class Variable:
    """ A generic Variable class to be used with decompiler"""

    def __init__(self, name, labels=[]):
        self.name = name
        self.labels = labels

    def binary_repr(self):
        return self.name


class StackVariable:
    """ A class for a variable on the stack to be used with decompiler """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def binary_repr(self):
        return self.name


class Constant:
    """ A class to define a variable as a constant term to be used with decompiler"""

    def __init__(self, value):
        self.value = value

    def binary_repr(self):
        return self.value


class FilterVariable:
    """ A variable class for filters to be used with decompiler """

    def __init__(self, base):
        if isinstance(base, (Constant, StackVariable)):
            self.type = "string"
            self.value = base.value
        else:
            self.type = "var"
            for k, v in base.__dict__.items():
                self.__dict__[k] = v

    def __str__(self):
        return f"{self.type} - {self.value}"


class BinaryOperation:
    def __init__(self, left, right, opname):
        self.left = left
        self.right = right
        self.opname = opname

    def binary_repr(self):
        return [self.left, self.right, self.opname]


class FunctionCall:
    def __init__(self, args, kwargs, name):
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def binary_repr(self):
        return self


class Decompiler:
    """
    This class translates python bytecode to a format the Query class can use.

    Note
    =========
    Typically this class is not used directly by backends.
    
    """

    def __init__(self, gen, plan, *args, **kwargs):
        """
        Parameters
        ----------
        gen  : Generator
            The generator to be decompiled
        plan : Plan
            Initial execution plan information
        """
        self.plan = plan
        self.gen = gen

        #: list of StackVariable: The initial stack is empty
        self.stack = []
        #: dict of function: Dictionary for supported opcodes
        self.opcodes = {
            "BINARY_ADD": self.binary_ops,
            "BINARY_AND": self.binary_ops,
            "BINARY_FLOOR_DIVIDE": self.binary_ops,
            "BINARY_LSHIFT": self.binary_ops,
            "BINARY_MATRIX_MULTIPLY": self.binary_ops,
            "BINARY_MODULO": self.binary_ops,
            "BINARY_MULTIPLY": self.binary_ops,
            "BINARY_OR": self.binary_ops,
            "BINARY_POWER": self.binary_ops,
            "BINARY_RSHIFT": self.binary_ops,
            "BINARY_SUBSCR": self.binary_subscr,
            "BINARY_SUBTRACT": self.binary_ops,
            "BINARY_TRUE_DIVIDE": self.binary_ops,
            "BINARY_XOR": self.binary_ops,
            "COMPARE_OP": self.compare_op,
            "CALL_FUNCTION": self.call_function,
            "CALL_FUNCTION_KW": self.call_function,
            "FOR_ITER": self.for_iter,
            "JUMP_ABSOLUTE": self.jump_absolute,
            "LOAD_ATTR": self.load_attr,
            "LOAD_CONST": self.load_const,
            "LOAD_DEREF": self.load_deref,
            "LOAD_FAST": self.load_fast,
            "LOAD_GLOBAL": self.load_global,
            "POP_JUMP_IF_FALSE": self.no_op,
            "POP_TOP": self.pop_top,
            "RETURN_VALUE": self.return_value,
            "STORE_FAST": self.store_fast,
            "UNPACK_SEQUENCE": self.unpack_sequence,
            "YIELD_VALUE": self.yield_value,
        }

    def decompile(self):
        """ Start handleling the generator bytecode"""
        bytecode = dis.Bytecode(self.gen)
        for instr in bytecode:
            op = self.opcodes.get(instr.opname, None)
            if op:
                op(instr)
            else:
                raise NotImplementedError(f"opname of {instr.opname} is not supported")
        return self.plan

    def for_iter(self, instr):
        """ Handle FOR_ITER instruction """
        stack_val = self.stack.pop()
        from .query import Query

        if type(stack_val.value) == type(iter("")):
            self.plan.metrics = ["".join(stack_val.value)]
        elif isinstance(stack_val.value, Query):
            self.plan.metrics = stack_val.value._generate_plan()
        else:
            raise TypeError(f"Unexpected type of iterable ({type(stack_val.value)}")

    def load_fast(self, instr):
        """ Handle LOAD_FAST instruction """
        if instr.argval.startswith("."):
            self.stack.append(StackVariable(instr.argval, self.gen.gi_frame.f_locals[instr.argval]))
        else:
            i = None
            for i, v in enumerate(self.plan.variables):
                if v.name == instr.argval:
                    break
            if i >= 0:
                self.stack.append(self.plan.variables[i])
            else:
                raise TypeError("Unexpected variable name")

    def no_op(self, instr):
        pass

    def store_fast(self, instr):
        """ Handle STORE_FAST instruction """
        self.plan.variables.append(Variable(instr.argval, labels=[]))

    def yield_value(self, instr):
        pass

    def load_attr(self, instr):
        """ Handle LOAD_ATTR instruction """
        stack_val = self.stack.pop()
        stack_val.labels.append(instr.argval)
        self.stack.append(stack_val)

    def load_const(self, instr):
        """ Handle LOAD_CONST instruction """
        self.stack.append(Constant(instr.argval))
        pass

    def compare_op(self, instr):
        """ Handle COMPARE_OP instruction """
        r = self.stack.pop()
        l = self.stack.pop()
        right = FilterVariable(r)
        left = FilterVariable(l)

        # need to clear out labels used for filtering
        r.labels = []
        l.labels = []

        self.plan.filters.append({"right": right, "left": left, "op": instr.argval})

    def load_global(self, instr):
        """ Handle LOAD_GLOBAL instruction """
        self.stack.append(
            StackVariable(instr.argval, self.gen.gi_frame.f_globals.get(instr.argval, "unknown"))
        )

    def load_deref(self, instr):
        """ Handle LOAD_DEREF instruction """
        self.stack.append(
            StackVariable(instr.argval, self.gen.gi_frame.f_locals.get(instr.argval, "unknown"))
        )

    def pop_top(self, instr):
        pass

    def jump_absolute(self, instr):
        pass

    def return_value(self, instr):
        pass

    def call_function(self, instr):
        kwargs = {}
        args = []
        arg_count = instr.arg
        if instr.opname == "CALL_FUNCTION_KW":
            kwarg_names = self.stack.pop()
            for n in reversed(kwarg_names.value):
                kwargs[n] = self.stack.pop()
                arg_count -= 1
        for _ in range(arg_count):
            args.insert(0, self.stack.pop())
        name = self.stack.pop().name
        func = FunctionCall(args, kwargs, name)
        if self.plan.calc:
            self.plan.calc.append(func)
        else:
            self.plan.calc = [func]
        self.stack.append(func)

    def binary_ops(self, instr):
        """ Handle binary operation instructions """
        binary_op = []
        right = self.stack.pop()
        left = None
        if len(self.stack) > 0:
            left = self.stack.pop().binary_repr()
        binary_op = BinaryOperation(left, right.binary_repr(), instr.opname)
        if self.plan.calc:
            self.plan.calc.append(binary_op)
        else:
            self.plan.calc = [binary_op]

    def binary_subscr(self, instr):
        """ Handle BINARY_SUBSCR instruction """
        index = self.stack.pop()
        target = self.stack.pop()

        self.stack.append(Constant(target.value[index.value]))

    def unpack_sequence(self, instr):
        pass
