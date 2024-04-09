from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='Post2p',
    version='1.0.0',
    packages=find_packages(),
    package_data={'': ['2pCodes/*']},
    url='https://github.com/faezehrabbani97/Post2p',
    license='MIT',
    author='Faezeh Rabbani',
    author_email='faezeh.rabbani97@gmail.com',
    description='Description',
    install_requires=[requirements],
)
