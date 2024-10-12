from setuptools import setup, find_packages

description_ = """Library that gives an abstraction layer to work with Alexnet CNN wit"""

setup(
    name='alexnet',  # Name of your library
    version='0.1.0',  # Initial version number
    author='Percy Cubas',  # Your name
    author_email ='pcubasm1@gmail.com',  # Your email address
    description = description_,  # Short description
    long_description=open('README.md').read(),  # Long description from README
    long_description_content_type='text/markdown',  # Format of long description
    url='https://github.com/pQbas/alexnet-pytorch.git',  # URL of your project's repository
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change as per your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the minimum Python version required
    install_requires=[
        'torch==2.2.1',
        'torchvision==0.17.1',
        'numpy',
        'tqdm',
        'colorlog',
        'rich'
    ],
)

