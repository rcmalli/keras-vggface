from setuptools import setup, find_packages
exec(open('keras_vggface/version.py').read())
setup(
    name='keras_vggface',
    version=__version__ + '_vddk-0.1',
    description='VGGFace implementation with Keras framework',
    url='https://github.com/DavidDoukhan/keras-vggface',
    author='Refik Can MALLI & David Doukhan',
    author_email="mallir@itu.edu.tr",
    license='MIT',
    keywords=['keras', 'vggface', 'deeplearning'],
    packages=find_packages(exclude=["tools", "training", "temp", "test", "data", "visualize","image",".venv",".github"]),
    zip_safe=False,
    install_requires=[
        'numpy>=1.9.1', 'scipy>=0.14', 'h5py', 'pillow', 'tensorflow',
        'six>=1.9.0', 'pyyaml', 'keras-applications'
    ],
    )
