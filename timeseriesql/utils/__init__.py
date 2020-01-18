# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.


# https://gist.github.com/adamnew123456/9218f99ba35da225ca11
# this can be replaced https://docs.python.org/3/library/functools.html#functools.singledispatchmethod 
# but it is only available in 3.8+
from collections import namedtuple
from functools import singledispatch, update_wrapper


def dispatchmethod(func):
    """
    This provides for a way to use ``functools.singledispatch`` inside of a class. It has the same
    basic interface that ``singledispatch`` does:
    
    >>> class A:
    ...     @dispatchmethod
    ...     def handle_message(self, message):
    ...         # Fallback code...
    ...         pass
    ...     @handle_message.register(int)
    ...     def _(self, message):
    ...         # Special int handling code...
    ...         pass
    ...
    >>> a = A()
    >>> a.handle_message(42)
    # Runs "Special int handling code..."
    
    Note that using ``singledispatch`` in these cases is impossible, since it tries to dispatch
    on the ``self`` argument, not the ``message`` argument. This is technically a double
    dispatch, since both the type of ``self`` and the type of the second argument are used to
    determine what function to call - for example:
    
    >>> class A:
    ...     @dispatchmethod
    ...     def handle_message(self, message):
    ...         print('A other', message)
    ...         pass
    ...     @handle_message.register(int)
    ...     def _(self, message):
    ...         print('A int', message)
    ...         pass
    ...
    >>> class B:
    ...     @dispatchmethod
    ...     def handle_message(self, message):
    ...         print('B other', message)
    ...         pass
    ...     @handle_message.register(int)
    ...     def _(self, message):
    ...         print('B int', message)
    ...
    >>> def do_stuff(A_or_B):
    ...     A_or_B.handle_message(42)
    ...     A_or_B.handle_message('not an int')
    
    On one hand, either the ``dispatchmethod`` defined in ``A`` or ``B`` is used depending
    upon what object one passes to ``do_stuff()``, but on the other hand, ``do_stuff()``
    causes different versions of the dispatchmethod (found in either ``A`` or ``B``) 
    to be called (both the fallback and the ``int`` versions are implicitly called).
    
    Note that this should be fully compatable with ``singledispatch`` in any other respects
    (that is, it exposes the same attributes and methods).
    """
    dispatcher = singledispatch(func)

    def register(type, func=None):
        if func is not None:
            return dispatcher.register(type, func)
        else:

            def _register(func):
                return dispatcher.register(type)(func)

            return _register

    def dispatch(type):
        return dispatcher.dispatch(type)

    def wrapper(inst, dispatch_data, *args, **kwargs):
        cls = type(dispatch_data)
        impl = dispatch(cls)
        return impl(inst, dispatch_data, *args, **kwargs)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = dispatcher.registry
    wrapper._clear_cache = dispatcher._clear_cache
    update_wrapper(wrapper, func)
    return wrapper
