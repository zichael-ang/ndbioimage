import pytest
from pathlib import Path
from ndbioimage import Imread, ReaderNotFoundError


@pytest.mark.parametrize("file", (Path(__file__).parent / 'files').iterdir())
def test_open(file):
    try:
        with Imread(file) as im:
            print(im[dict(c=0, z=0, t=0)].mean())
    except ReaderNotFoundError:
        assert len(Imread.__subclasses__()), "No subclasses for Imread found."
