from setuptools import setup, find_packages

setup(name='keras_vggface',
      version='0.2',
      description='VGGFace implementation with Keras framework',
      url='https://github.com/rcmalli/keras-vggface',
      author='Refik Can MALLI',
      author_email = "mallir@itu.edu.tr",
      license='MIT',
      keywords = ['keras', 'vggface'],
      packages=find_packages(exclude=["temp", "image", "test"]),
      zip_safe=False,
      install_requires=['numpy', 'pillow', 'tensorflow', 'keras'])