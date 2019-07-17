from setuptools import setup, find_packages
exec(open('keras_vggface/version.py').read())
setup(
    name='keras_vggface',
    version=__version__,
    description='VGGFace implementation with Keras framework',
    url='https://github.com/rcmalli/keras-vggface',
    author='Refik Can MALLI',
    author_email="mallir@itu.edu.tr",
    license='MIT',
    keywords=['keras', 'vggface', 'deeplearning'],
    packages=find_packages(exclude=["tools", "training", "temp", "test", "data", "visualize","image",".venv",".github"]),
    zip_safe=False,
    install_requires=[
        'numpy>=1.9.1', 'scipy>=0.14', 'h5py', 'pillow', 'keras',
        'six>=1.9.0', 'pyyaml'
    ],
    extras_require={
        "tf": ["tensorflow"],
        "tf_gpu": ["tensorflow-gpu"],
    })
