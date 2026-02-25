import setuptools

setuptools.setup(name='stead-ai',
                 version='1.0',
                 author='Xihao Hu',
                 author_email='huxihao@gmail.com',
                 packages=[''],
                 install_requires=[
                     'torch', 'torchvision', 'matplotlib', 'pandas', 'scipy',
                     'scikit-learn', 'seaborn', 'ipython', 'jupyter', 'lifelines',
                     'pyarrow', 'fastparquet', 'dask'
                 ],
                 zip_safe=False)
