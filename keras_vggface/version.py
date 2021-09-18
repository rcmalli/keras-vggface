__version__ = '0.6'

def pretty_versions():
    import tensorflow.keras
    import tensorflow as tf
    k_version = tensorflow.keras.__version__
    t_version = tf.__version__
    return "keras-vggface : {}, keras : {} , tensorflow : {} ".format(__version__,k_version,t_version)
