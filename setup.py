from setuptools import setup

setup(
    name='icasso',
    version='0.1.0',
    description="",
    author='Erkka Heinila',
    author_email='erkka.heinila@jyu.fi',
    url='https://github.com/Teekuningas/icasso',
    license='BSD',
    packages=['icasso'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'setuptools',
        'numpy',
        'matplotlib',
        'sklearn',
        'scipy',
    ],
)

