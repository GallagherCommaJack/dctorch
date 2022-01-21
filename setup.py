from setuptools import setup, find_packages

setup(
    name='dctorch',
    version='0.1.0',
    description='2d dct using axial matmul with cached matrices',
    url='https://github.com/GallagherCommaJack/dctorch',
    author='Jack Gallagher',
    author_email='jack@gallabytes.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.22',
        'scipy>=1.7',
        'torch>=1.10',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
