import setuptools
import platform
import os

version = '2022.7.0'

if platform.system().lower() == 'linux':
    import pkg_resources
    pkg_resources.require(['pip >= 20.3'])

with open('README.md', 'r') as fh:
    long_description = fh.read()


with open(os.path.join(os.path.dirname(__file__), 'ndbioimage', '_version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")
    try:
        with open(os.path.join(os.path.dirname(__file__), '.git', 'refs', 'heads', 'master')) as h:
            f.write("__git_commit_hash__ = '{}'\n".format(h.read().rstrip('\n')))
    except Exception:
        f.write(f"__git_commit_hash__ = 'unknown'\n")


setuptools.setup(
    name='ndbioimage',
    version=version,
    author='Wim Pomp',
    author_email='w.pomp@nki.nl',
    description='Bio image reading, metadata and some affine registration.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wimpomp/ndbioimage',
    packages=['ndbioimage', 'ndbioimage.readers'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=['untangle', 'pandas', 'psutil', 'numpy', 'tqdm', 'tifffile', 'czifile', 'pyyaml', 'dill',
                      'tiffwrite'],
    extras_require={'transforms': 'SimpleITK-SimpleElastix',
                    'bioformats': ['python-javabridge', 'python-bioformats']},
    tests_require=['pytest-xdist'],
    entry_points={'console_scripts': ['ndbioimage=ndbioimage.ndbioimage:main']},
    package_data={'': ['transform.txt']},
    include_package_data=True,
)
