from setuptools import setup, find_packages

with open('README.txt') as f:
    long_description = f.read()

setup(name='ace',
    version='0.2',
    description='Non-parametric multivariate regressions by Alternating Conditional Expectations',
    author='Nick Touran',
    author_email='ace@partofthething.com',
    url='https://github.com/partofthething/ace',
    packages=find_packages(),
    license='MIT',
    long_description=long_description,
    install_requires=['numpy', 'scipy'],
    keywords='regression ace multivariate statistics',
    use_2to3=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        ],
      test_suite='tests'

     )
