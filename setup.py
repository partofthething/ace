from setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='ace',
    version='0.3.2',
    description=
    'Non-parametric multivariate regressions by Alternating Conditional Expectations',
    author='Nick Touran',
    author_email='ace@partofthething.com',
    url='https://github.com/partofthething/ace',
    packages=find_packages(),
    license='MIT',
    long_description=long_description,
    install_requires=['numpy', 'scipy>=0.17'],
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests')
