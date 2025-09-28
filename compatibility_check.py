import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) < version.parse("2.15.0"):
    print("TF version too old. Use version 2.15.0")
elif version.parse(tf.__version__) > version.parse("2.15.0"):
    print("TF version doesnot support. Use version 2.15.0")
else:
    print("TF version is compatible")
