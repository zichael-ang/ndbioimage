import pickle
from multiprocessing import active_children
from pathlib import Path

import pytest
from ndbioimage import Imread, ReaderNotFoundError


@pytest.mark.parametrize('file', (Path(__file__).parent / 'files').iterdir())
def test_open(file):
    try:
        with Imread(file) as im:
            mean = im[dict(c=0, z=0, t=0)].mean()
            b = pickle.dumps(im)
            jm = pickle.loads(b)
            assert jm[dict(c=0, z=0, t=0)].mean() == mean
            v = im.view()
            assert v[dict(c=0, z=0, t=0)].mean() == mean
            b = pickle.dumps(v)
            w = pickle.loads(b)
            assert w[dict(c=0, z=0, t=0)].mean() == mean
    except ReaderNotFoundError:
        assert len(Imread.__subclasses__()), 'No subclasses for Imread found.'

    for child in active_children():
        child.kill()
