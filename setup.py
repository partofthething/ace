from distutils.core import setup

setup(name='ace',
    version='0.1',
    description='Alternating Conditional Expectations',
    author='Nick Touran',
    author_email='ace@partofthething.com',
    url='https://github.com/partofthething/ace',
    packages=['ace'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research'
        'Topic :: Scientific/Engineering :: Information Analysis'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',

        ],
      install_requires=['numpy'],
     )