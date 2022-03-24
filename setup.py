from setuptools import setup

setup(name='xtracker',
      version='0.1.0',
      description='Neural network based trackfinding for Belle II experiment at SuperKEKB',
      long_description=open('README.md').read(),
      classifiers=["License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"],
      keywords='tracking graph neural networks',
      url='https://github.com/BenjaminSchwenker/xtracker',
      author='Benjamin Schwenker',
      author_email='benjamin.schwenker@phys.uni-goettingen.de',
      license='GNU General Public License v3 or later (GPLv3+)',
      packages=['xtracker'],
      install_requires=[],
      include_package_data=True,
      zip_safe=False)
