import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras

tf.experimental.numpy.experimental_enable_numpy_behavior()

class KalmanNet(keras.Model):
  def __init__(self, f, h, n, m, dt=1e-2):
    super().__init__()
    self.f = f
    self.h = h
    self.n = n
    self.m = m
    self.dt = dt

    self.x1 = tf.zeros((1, n))
    self.x2 = tf.zeros((1, n))
    self.gQ   = tf.zeros((1, n*n))
    self.gSxx = tf.zeros((1, n*n))
    self.gSyy   = tf.zeros((1, m*m))

    self.gru_Q = keras.layers.GRUCell(n*n)
    self.gru_Syy = keras.layers.GRUCell(m*m)

    self.dense_Y = keras.layers.Dense(10, activation='relu')

    self.dense_Sxx_in = keras.layers.Dense(10, activation='relu')
    self.dense_Sxx = keras.layers.Dense(n*n)
    self.dense_Sxx_out = keras.layers.Dense(5, activation='relu')

    self.dense_Sxy_in = keras.layers.Dense(10, activation='relu')
    self.dense_Sxy = keras.layers.Dense(n*m)

    self.dense_1 = keras.layers.Dense(30, activation='relu')
    self.dense_2 = keras.layers.Dense(n*n, activation='relu')
    

  def call(self, inputs, training=False):
    nbatch = inputs.shape[0]
    tsteps = inputs.shape[1]

    n = self.n
    m = self.m
    f = self.f
    h = self.h
    dt = self.dt

    gQ   = self.gQ
    gSxx = self.gSxx
    gSyy = self.gSyy
    x1 = self.x1
    x2 = self.x2

    y1 = tf.zeros((1, m))

    X_out = tf.zeros((1, tsteps, n))

    r2 = scipy.integrate.ode(f).set_integrator('dopri5')
    r2.set_initial_value(np.squeeze(x2), r2.t)

    r1 = scipy.integrate.ode(f).set_integrator('dopri5')
    r1.set_initial_value(np.squeeze(x1), r1.t)

    for t in range(tsteps):
      y = inputs[:, t]

      r2.integrate(r2.t + dt)
      E_x2 = np.expand_dims(r2.y, axis=0)

      del_x_hat = x1 - E_x2
      del_x_tilde = x1 - x2

      r1.integrate(r1.t + dt)
      E_x1 = np.expand_dims(r1.y, axis=0)

      del_y_hat = (y - h(E_x1)).astype(float)
      del_y_tilde = y - y1

      p = del_x_hat
      Q, gQ = self.gru_Q(p, gQ)
      Q = tf.reshape(Q, (1, n, n))
      Q = tf.reshape(Q @ tf.transpose(Q, perm=(0, 2, 1)), (1, n*n))

      p = del_x_tilde.astype(float)
      p = tf.concat((Q, p), axis=1)
      s = tf.concat((p, gSxx), axis=1)
      s = self.dense_Sxx_in(s)
      Sxx = self.dense_Sxx(s)

      p = tf.concat((del_y_hat, del_y_tilde), axis=1)
      p = self.dense_Y(p)

      pSxx = self.dense_Sxx_out(Sxx)
      p = tf.concat((p, pSxx), axis=1)
      invSyy, gSyy = self.gru_Syy(p, gSyy)

      p = tf.concat((Sxx, invSyy), axis=1)
      p = self.dense_Sxy_in(p)
      Sxy = self.dense_Sxy(p)

      m_Sxy = tf.reshape(Sxy, (1, n, m))
      m_invSyy = tf.reshape(invSyy, (1, m, m))
      m_invSyy = m_invSyy @ tf.transpose(m_invSyy, perm=(0, 2, 1))
      K = tf.reshape(m_Sxy @ m_invSyy, (1, n*m))

      p = tf.concat((invSyy, K), axis=1)
      p = self.dense_1(p)
      p = tf.concat((p, Sxx), axis=1)
      gSxx = self.dense_2(p)

      Kg = tf.reshape(K, (nbatch, n, m))

      x2 = tf.identity(x1)
      del_y_hat = tf.expand_dims(del_y_hat, axis=2)
      x1 = E_x1 + tf.squeeze(Kg@del_y_hat, axis=2)

      y1 = tf.identity(y)

      X_out += tf.scatter_nd([[0, t]], x1, [nbatch, tsteps, n])

    if(not training):
      self.gQ = gQ
      self.gSxx = gSxx
      self.gSyy = gSyy
      self.x1 = x1
      self.x2 = x2

    return X_out