# MNIST dataset example
# Why Tensorflow? It allows cheaper array and matrix operations
# outside Python than Python itself (instead of a single one, it performs several)
# Tensorflow uses graphs to achieve this. TensorFlow isn't a rigid neural networks library. 
# If you can express your computation as a data flow graph, you can use TensorFlow. 


#state of the art in classification of objects (for various datasets)
#http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

import os
os.chdir('/home/sophie/tensorflow_tutorial')
import tensorflow as tf

#Reading data automatically from website
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#55K imagenes de 28X28=784 pixeles c/u. Cada entrada de los 784 pixeles 
#representa una intensidad (0,1). Se dice que cada una de las imagenes son un tensor
mnist.train.images.shape

#55K labels del tipo one-hot vectors del 1-9, i.e. 55K vectores de dimension 10. Tambien las
##labels son un tensor
## Me dice que digito tiene cada imagen
##Un tensor simplemente es un arreglo n-dimensional 
mnist.train.labels.shape

##softmax gives us a list of values between 0 and 1 that add up to 1
## Perfect to model class probabilities, i.e. probability of belonging to a class
# Usually when we want to model this, the last layer will be a softmax layer
 
# Softmax regression has 2 steps: 1. Evidence of belonging to each class
# #2. Convert the evidence to probabilities
# 
# To get the evidence of an image being is of certain class, we do a weighted sum of
# every pixel in the image: we have to learn negative and positive weights for each pixel, per class
# 
# So we end up with some evidence =Wx+bias
# We convert it to probabilities: y=softmax(evidence)

#################################################
##############Variables definition###############
#################################################
####PLACEHOLDER (tensor): a place to run computations
# Tensorflow symbolic variables creation. We wont use this until we ask to run a computation
# 2-D tensor of floating-point numbers, with a shape [None, 784]
# None means the dimension can be of any shape 
x = tf.placeholder(tf.float32, [None, 784])

####VARIABLE (tensor): modifiable tensor that lives in TensorFlow's graph of interacting computations
#It can be used or modified by the computation, so usually these are the parameters of the model
# Inicializamos las variables en cero

#vamos a multiplicar una imagen por W para tener la evidencia de pertenencia a cada clase
W = tf.Variable(tf.zeros([784, 10]))

# b tiene dimensión de 10 para sumárselo al output
b = tf.Variable(tf.zeros([10]))

#################################################
#################Model definition################
#################################################
y = tf.nn.softmax(tf.matmul(x, W) + b)

#################################################
###################Model training################
#################################################
#We define the loss using cross-entropy: how inefficient our predictions are from the truth

# Placeholder to put the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# -------Loss:------ We define cross-entropy loss
# tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the
# corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, 
# due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# -----Optimization:---- we ask Tensorflow to minimixe cross_entropy using Gradent descent with
#a learning rate of 0.5
#tensorflow is ADDING new  operations to our graph which implements backpropagation	
#and gradient descent
# it gives you back a single operation, which, when run, does a step of gradient descent
#training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#-------Setting up model to train: ------
#create an operation to initialize the variables we created.
init = tf.global_variables_initializer()
#We can launch the model in a session and run the operation that initializes the variables
sess = tf.Session()
sess.run(init)

#-------Train!-------
# in each step of the loop we have a batch of one hundred random data points
# from our training set. 
# We run train_step feeding in the batches data to replace the placeholders.
# using small batches of random data is called stochastic training: stochastic gradient descent
# using all data is expensive, instead we use a different subset

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#################################################
###################Model evaluation##############
#################################################
# tf.argmax gives us the index with the highest entry in a tensor along some axis
# tf.argmax(y,1) is the label our model thinks is most likely for each input 
# tf.argmax(y_,1) is the correct label 
# tf.equal to check if our prediction matches the truth, it gives a list ob booleans
 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#what fraction are correct, we cast to floating point numbers and then take the mean. 
#For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# we print our accuracy on our test data: 92%, which is bad!
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))










