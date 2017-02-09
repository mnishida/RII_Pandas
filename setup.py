from os import path
from setuptools import setup
import riip

here = path.abspath(path.dirname(__file__))

# Get the long description from the RpythonEADME file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='riip',
      version=riip.__version__,
      description=('Python 3 + Pandas wrapper ' +
                   'for the refractiveindex.info database.'),
      long_description=long_description,
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      keywords='refractive index, dielectric constant, optical material',
      author=riip.__author__,
      author_email='mnishida@hiroshima-u.ac.jp',
      url='https://github.com/mnishida/Riip',
      license=riip.__license__,
      packages=['riip', 'tests', 'examples'],
      include_package_data=True,
      data_files=[
          # ('data',
          # [path.join('data', 'catalog.csv'),
          # path.join('data', 'grid_data.csv'),
          # path.join('data', 'raw_data.csv')]),
          ('examples', [path.join('examples', 'examples.ipynb')])],
      zip_safe=False,
      install_requires=[
          'setuptools',
          'numpy',
          'scipy',
          'pandas',
          'pyyaml',
          'gitpython',
          'matplotlib'
      ],
      entry_points="""
      # -*- Entry points: -*-
      """
      )
