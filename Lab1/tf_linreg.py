import tensorflow as tf

# 1. defining calculation graph
# data and parameters
X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

n = tf.shape(X, out_type=tf.int32)[0]

# affine regression model
Y = a * X + b

# square loss
loss = (Y-Y_)**2/(2*tf.to_float(n))

# calculation of gradients
grad_a = tf.reduce_sum(-X * (Y_- Y)) / (tf.to_float(n))
grad_b = tf.reduce_sum(-(Y_- Y)) / (tf.to_float(n))

# printing calculated gradients with tf every time they are calculated
grad_a = tf.Print(grad_a, [grad_a])
grad_b = tf.Print(grad_b, [grad_b])

# optimisation: gradient descent
trainer = tf.train.GradientDescentOptimizer(0.1)

# computing and applying gradients manually
grads_and_vars = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(grads_and_vars)

# 2. init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. learning
for i in range(100):
    val_loss, _, val_a, val_b, val_grads_and_vars, val_grad_a, val_grad_b = \
        sess.run([loss, train_op, a, b, grads_and_vars, grad_a, grad_b],
                 feed_dict={X: [1,2,3,4,5], Y_: [3,5,7,9,11]})
    print(i, val_loss, val_a, val_b)

    for gradient, var in val_grads_and_vars:
        print(gradient)

    print(val_grad_a)
    print(val_grad_b)


