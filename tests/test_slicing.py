from itertools import combinations_with_replacement
from numbers import Number

import numpy as np
import pytest

from ndbioimage import Imread

r = np.random.randint(0, 255, (64, 64, 2, 3, 4))
im = Imread(r)
a = np.array(im)


@pytest.mark.parametrize('s', combinations_with_replacement(
    (0, -1, 1, slice(None), slice(0, 1), slice(-1, 0), slice(1, 1)), 5))
def test_slicing(s):
    s_im, s_a = im[s], a[s]
    if isinstance(s_a, Number):
        assert isinstance(s_im, Number)
        assert s_im == s_a
    else:
        assert isinstance(s_im, Imread)
        assert s_im.shape == s_a.shape
        assert np.all(s_im == s_a)
