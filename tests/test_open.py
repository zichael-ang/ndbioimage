import pytest
from pathlib import Path
from ndbioimage import Imread


@pytest.mark.parametrize("file",
                         [file for file in (Path(__file__).parent / 'files').iterdir() if file.suffix != '.pzl'])
def test_open(file):
    with Imread(file) as im:
        print(im[dict(c=0, z=0, t=0)].mean())
