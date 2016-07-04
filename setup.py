from setuptools import setup, find_packages

setup(
    name='fat-walrus',
    version='0.0.1',
    author='Aaron Schein',
    author_email='aaron.j.schein@gmail.com',
    description=('A Python class for generative models'),
    license='None yet',
    keywords='walrus leopard seal sea lion aquatic mammal fat',
    url='https://github.com/aschein/fat-walrus',
    packages=find_packages(),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'seaborn'],
    tests_require = ['nose'],
)
