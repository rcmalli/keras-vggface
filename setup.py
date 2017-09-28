from setuptools import setup, find_packages

setup(name='keras_vggface',
      version='0.4',
      description='VGGFace implementation with Keras framework',
      url='https://github.com/rcmalli/keras-vggface',
      author='Refik Can MALLI',
      author_email = "mallir@itu.edu.tr",
      license='MIT',
      keywords = ['keras', 'vggface', 'deeplearning'],
      packages=find_packages(exclude=["temp", "test"]),
      zip_safe=False,
      install_requires=['numpy', 'pillow', 'tensorflow', 'keras', 'h5py'])