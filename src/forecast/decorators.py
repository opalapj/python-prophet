from copy import deepcopy
from functools import wraps

from nercast.classes import Model


def _work_on_copy(fun):
    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        # Preserving references to the originals and creating copies for
        # further modifications.
        orig_obj, self._obj = self._obj, self._obj.copy()
        orig_model, self.model = self.model, deepcopy(self.model)
        # Instantiation of a new model object, if it has not existed so far.
        self.model = self.model or Model()
        # Calling decorated function.
        obj = fun(self, *args, **kwargs)
        # Checking whether the newly created model instance has been modified.
        if self.model != Model():
            # Accessor is instantiated during first call on newly created
            # DataFrame. Due to this fact, newly created and modified
            # model has to be attributed just after creation.
            obj.fcst.model = self.model
        else:
            obj.fcst.model = orig_model
        self._obj = orig_obj
        self.model = orig_model
        return obj

    return wrapper
