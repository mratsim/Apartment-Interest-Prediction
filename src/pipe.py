# inspired by: https://joshbohde.com/blog/functional-python
# Transformations do not take extra arguments so no need for partial or starmap

from functools import reduce

#compose list of functions (chained composition)
def compose(*funcs):
    def _compose(f, g):
        # functions are expecting X,y not (X,y) so must unpack with *g
        return lambda *args, **kwargs: f(*g(*args, **kwargs))
    return reduce(_compose, funcs)

# pipe functions, reverse the order so that it's in the usual FIFO function order
def pipe(*funcs):
    return compose(*reversed(funcs))