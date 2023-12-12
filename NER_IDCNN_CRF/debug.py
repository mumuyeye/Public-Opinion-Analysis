import tensorflow as tf
 
tensorflow_version = tf.__version__
 
#以下两行代码适合有“布置GPU环境的”
gpu_available = tf.test.is_gpu_available()
print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)
 
#以下一行代码适合没有“布置GPU环境的”，纯CPU版本的
#print("tensorflow version:", tensorflow_version)
 
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(result)