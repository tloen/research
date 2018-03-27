import oesort as odd_even
import numpy as numpy
import tensorflow as tf
import random

def tf_smooth_maximum(x1, x2, t = 1):
  temp1, temp2 = tf.exp(tf.divide(x1, t)), tf.exp(tf.divide(x2, t))
  return tf.multiply(tf.log(temp1 + temp2 - 1), t)

SIGMA = tf.constant(5.0, dtype = 'float64')

def tf_smooth_exchange(x1, r1, x2, r2, t = 1):
  sm = tf.add(x1, x2)
  smax = tf_smooth_maximum(x1, x2, t)
  smin = tf.subtract(sm, smax)
  # denom = tf.subtract(x1, x2)
  avg = tf.divide(sm, 2)
  aff1 = tf.sigmoid(tf.multiply(SIGMA, tf.divide(tf.subtract(smax, avg), tf.subtract(x1, x2))))
  aff2 = 1 - aff1
  rmax = tf.add(tf.multiply(aff1, r1), tf.multiply(aff2, r2))
  rmin = tf.add(tf.multiply(aff1, r2), tf.multiply(aff2, r1))
  return smax, rmax, smin, rmin


L = 2

r_in = [tf.placeholder(dtype='float64') for _ in range(L)]
s_in = [tf.placeholder(dtype='float64') for _ in range(L)]
r, s = r_in[:], s_in[:]

pairs = odd_even.oddeven_merge_sort(L)
for p in pairs:
  n, m = p
  wires = s[n], r[n], s[m], r[m]
  s[n], r[n], s[m], r[m] = tf_smooth_exchange(*wires, t=.5)

with tf.Session() as sess:
  feedr = [(r_in[i], random.randint(0, 20)) for i in range(L)]
  feeds = [(s_in[i], random.randint(0, 20)) for i in range(L)]
  feed = dict(feedr + feeds)
  for i in range(L):
    print(sess.run([r[i], s[i]], feed_dict = feed))


