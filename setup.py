from setuptools import setup

setup(
   name='datasets',
   version='1.0',
   description='Dataset module for importing dataset parsers',
   author='Sushrut Bhalla',
   author_email='sushrut.bhalla@example.com',
   packages=['datasets'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
