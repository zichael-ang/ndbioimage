import pytest
import numpy as np
from ndbioimage import Imread
from itertools import product


r = np.random.randint(0, 255, (64, 64, 2, 3, 4))
im = Imread(r)
a = np.array(im)


@pytest.mark.parametrize('fun_and_axis', product(
    (np.sum, np.nansum, np.min, np.nanmin, np.max, np.nanmax, np.argmin, np.argmax,
     np.mean, np.nanmean, np.var, np.nanvar, np.std, np.nanstd), (None, 0, 1, 2, 3, 4)))
def test_ufuncs(fun_and_axis):
    fun, axis = fun_and_axis
    assert np.all(np.isclose(fun(im, axis), fun(a, axis))), \
        f'function {fun.__name__} over axis {axis} does not give the correct result'
