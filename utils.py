import os
import sys
import json
import tensorflow as tf
import numpy as np
import traceback
import math

_EPS = np.finfo(float).eps * 4.0

def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

def softzeroclip(x, slopeatzero=0.25):
  upp = 0.99
  up = (upp - 0.5) / slopeatzero
  cond1 = tf.cast(tf.math.less(x, 0.0), tf.float32)
  cond2 = tf.cast(tf.math.logical_and(tf.math.less(x, up), tf.math.greater_equal(x, 0.0)), tf.float32)
  cond3 = tf.cast(tf.math.greater_equal(x, up), tf.float32)

  a = tf.math.multiply(cond1, tf.sigmoid(x * slopeatzero / 0.25))
  b = tf.math.multiply(cond2, slopeatzero * x + 0.5)
  c = tf.math.multiply(cond3, tf.tanh((x-up) * slopeatzero/(1-upp))*(1-upp) + upp)
  return a + b + c

def get_pixel_value(img, u, v):
    indices = tf.stack([ u, v], -1)
    return tf.gather_nd(img, indices)

def get_pixel_value2(img, u, v):
    batch_idx = tf.range(0, tf.shape(img)[0])
    batch_idx = tf.reshape(batch_idx, (tf.shape(img)[0], 1, 1))
    b = tf.tile(batch_idx, (1, tf.shape(u)[1], tf.shape(u)[2]))
    indices = tf.stack([b, u, v], -1)
    return tf.gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    # x is in range 0..W
    # y is in range 0..H
    H = int(img.get_shape()[0])
    W = int(img.get_shape()[1])
    max_y = tf.cast(H-1, tf.int32)
    max_x = tf.cast(W-1, tf.int32)
    zero = tf.zeros([], dtype='int32')

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    Ia = get_pixel_value(img, y0, x0)
    Ib = get_pixel_value(img, y0, x1)
    Ic = get_pixel_value(img, y1, x0)
    Id = get_pixel_value(img, y1, x1)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    wa = (y1-y)*(x1-x)
    wb = (y1-y)*(x-x0)
    wc = (y-y0)*(x1-x)
    wd = (y-y0)*(x-x0)

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    out = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return out

def bilinear_sampler2(img, x, y):
    # x is in range 0..W
    # y is in range 0..H
    H = int(img.get_shape()[0])
    W = int(img.get_shape()[1])
    max_y = tf.cast(H-1, tf.int32)
    max_x = tf.cast(W-1, tf.int32)
    zero = tf.zeros([], dtype='int32')

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    Ia = get_pixel_value2(img, y0, x0)
    Ib = get_pixel_value2(img, y0, x1)
    Ic = get_pixel_value2(img, y1, x0)
    Id = get_pixel_value2(img, y1, x1)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    wa = (y1-y)*(x1-x)
    wb = (y1-y)*(x-x0)
    wc = (y-y0)*(x1-x)
    wd = (y-y0)*(x-x0)

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    out = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return out

def get_pixel_value3d(mpi, y, x, z):
    indices = tf.stack([z, y, x], -1)
    return tf.gather_nd(mpi, indices)

def linear3d_sampler(mpi, x, y, z):
  D = int(mpi.get_shape()[0])
  H = int(mpi.get_shape()[1])
  W = int(mpi.get_shape()[2])
  max_y = tf.cast(H-1, tf.int32)
  max_x = tf.cast(W-1, tf.int32)
  max_z = tf.cast(D-1, tf.int32)
  zero = tf.zeros([], dtype='int32')

  x = ((x)*tf.cast(max_x-1, tf.float32))
  y = ((y)*tf.cast(max_y-1, tf.float32))
  z = (z * tf.cast(max_z-1, tf.float32))

  x0 = tf.cast(tf.floor(x), tf.int32)
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), tf.int32)
  y1 = y0 + 1
  z0 = tf.cast(tf.floor(z), tf.int32)
  z1 = z0 + 1

  x0 = tf.clip_by_value(x0, zero, max_x)
  x1 = tf.clip_by_value(x1, zero, max_x)
  y0 = tf.clip_by_value(y0, zero, max_y)
  y1 = tf.clip_by_value(y1, zero, max_y)
  z0 = tf.clip_by_value(z0, zero, max_z)
  z1 = tf.clip_by_value(z1, zero, max_z)

  Ia = get_pixel_value3d(mpi, y0, x0, z0)
  Ib = get_pixel_value3d(mpi, y0, x1, z0)
  Ic = get_pixel_value3d(mpi, y1, x0, z0)
  Id = get_pixel_value3d(mpi, y1, x1, z0)


  Ia1 = get_pixel_value3d(mpi, y0, x0, z1)
  Ib1 = get_pixel_value3d(mpi, y0, x1, z1)
  Ic1 = get_pixel_value3d(mpi, y1, x0, z1)
  Id1 = get_pixel_value3d(mpi, y1, x1, z1)

  x0 = tf.cast(x0, tf.float32)
  x1 = tf.cast(x1, tf.float32)
  y0 = tf.cast(y0, tf.float32)
  y1 = tf.cast(y1, tf.float32)
  z0 = tf.cast(z0, tf.float32)
  z1 = tf.cast(z1, tf.float32)

  wa = (y1-y)*(x1-x)
  wb = (y1-y)*(x-x0)
  wc = (y-y0)*(x1-x)
  wd = (y-y0)*(x-x0)

  wa = tf.expand_dims(wa, axis=-1)
  wb = tf.expand_dims(wb, axis=-1)
  wc = tf.expand_dims(wc, axis=-1)
  wd = tf.expand_dims(wd, axis=-1)
  z1z= tf.expand_dims(z1-z, axis=-1)
  z0z= tf.expand_dims(z-z0, axis=-1)

  out = wa*Ia + wb*Ib + wc*Ic + wd*Id
  out *= z1z
  out += z0z*(wa*Ia1 + wb*Ib1 + wc*Ic1 + wd*Id1)

  return out

def nearest3d_sampler(mpi, x, y, z):
  D = int(mpi.get_shape()[0])
  H = int(mpi.get_shape()[1])
  W = int(mpi.get_shape()[2])
  max_y = tf.cast(H-1, tf.int32)
  max_x = tf.cast(W-1, tf.int32)
  max_z = tf.cast(D-1, tf.int32)
  zero = tf.zeros([], dtype='int32')

  x = ((x)*tf.cast(max_x-1, tf.float32))
  y = ((y)*tf.cast(max_y-1, tf.float32))
  z = (z * tf.cast(max_z-1, tf.float32))

  x0 = tf.cast(tf.floor(x), tf.int32)
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), tf.int32)
  y1 = y0 + 1
  zz = tf.cast(tf.round(z), tf.int32)

  x0 = tf.clip_by_value(x0, zero, max_x)
  x1 = tf.clip_by_value(x1, zero, max_x)
  y0 = tf.clip_by_value(y0, zero, max_y)
  y1 = tf.clip_by_value(y1, zero, max_y)
  zz = tf.clip_by_value(zz, zero, max_z)

  Ia = get_pixel_value3d(mpi, y0, x0, zz)
  Ib = get_pixel_value3d(mpi, y0, x1, zz)
  Ic = get_pixel_value3d(mpi, y1, x0, zz)
  Id = get_pixel_value3d(mpi, y1, x1, zz)

  x0 = tf.cast(x0, tf.float32)
  x1 = tf.cast(x1, tf.float32)
  y0 = tf.cast(y0, tf.float32)
  y1 = tf.cast(y1, tf.float32)
  zz = tf.cast(zz, tf.float32)

  wa = (y1-y)*(x1-x)
  wb = (y1-y)*(x-x0)
  wc = (y-y0)*(x1-x)
  wd = (y-y0)*(x-x0)

  wa = tf.expand_dims(wa, axis=-1)
  wb = tf.expand_dims(wb, axis=-1)
  wc = tf.expand_dims(wc, axis=-1)
  wd = tf.expand_dims(wd, axis=-1)

  out = wa*Ia + wb*Ib + wc*Ic + wd*Id
  return out


def quaternion_from_matrix(matrix):
    matrix = np.r_[np.c_[matrix, np.array([[0],[0],[0]])], np.array([[0,0,0,1]])]
    # Return quaternion from rotation matrix.
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def quaternion_matrix(quaternion):
    # Return homogeneous rotation matrix from quaternion.
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([ #print(len(js))
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def unit_vector(data, axis=None, out=None):
    # Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    data = np.array(data, dtype=np.float64, copy=True)
    data /= math.sqrt(np.dot(data, data))
    return data

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    # Return spherical linear interpolation between two quaternions.
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def interpolate_rotation(m1, m2, t):
  q1 = quaternion_from_matrix(m1)
  q2 = quaternion_from_matrix(m2)
  return quaternion_matrix(quaternion_slerp(q1, q2, t))[:3, :3]


def findCameraSfm(dataset):
  path = "datasets/" + dataset + "/MeshroomCache/StructureFromMotion/"
  dr = os.listdir(path)
  if len(dr) == 0: return ""
  return path + dr[0] + "/cameras.sfm"

def findExrs(dataset):
  path = "datasets/" + dataset + "/MeshroomCache/PrepareDenseScene/"
  dr = os.listdir(path)
  if len(dr) == 0: return ""
  return path + dr[0]

def colored_hook(home_dir):
  """Colorizes python's error message.
  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    #f = f[:, :, np.newaxis, np.newaxis]
    f = f[:, :, np.newaxis, np.newaxis]
    #f = np.tile(f, [1, 1, int(x.shape[1]), 1])
    f = np.tile(f, [1, 1, int(x.shape[3]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format=None)
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1]* factor, s[2] * factor, s[3] ])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID')

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

def coarse2fine(mpi,lod_in,logres):
  mpis = []
  for i in range(logres):
      mpis.append(mpi)
      mpi = downscale2d(mpi)
  
  for i in range(logres):
      mpi = upscale2d(mpi)
      mpi = lerp_clip(mpis[-1-i],mpi,i+lod_in-(logres-1))
  return mpi