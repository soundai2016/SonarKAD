from setuptools import setup, find_packages
import os

with open('version.txt', 'r') as version_file:
    version = version_file.read().strip()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

extra_files = package_files('sonarkad')

setup(
    name="SonarKAD",
    version=version,
    url="https://soundai.com",
    author="Chen Xiaoliang",
    author_email="chenxiaoliang@soundai.com",
    description="Sonar Kolmogorov-Arnold Decomposition Based on GPU or CPU.",
    license="Apache License",    
    packages=find_packages(),
    include_package_data=True,
    package_data={'': extra_files},
    keywords=[
        "SonarKAD",      
        "Sonar Kolmogorov-Arnold Decomposition",        
        "sonar",
        "kan",
    ], 
    install_requires=[
        'numpy',
        'PyYAML',
        'torch',
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",        
    ],
)