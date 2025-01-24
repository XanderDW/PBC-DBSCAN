from setuptools import setup, find_packages

setup(
    name='dbscan_pbc',  
    version='0.1',      
    description='DBSCAN with Periodic Boundary Conditions', 
    author='X. de Wit', 
    author_email='x.m.d.wit@tue.nl', 
    url='https://github.com/XanderDW/PBC-DBSCAN',  
    packages=find_packages(),  
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

