import os
import sys
import json
import tensorflow as tf
import numpy as np
import traceback

def get_pixel_value(img, u, v):
    indices = tf.stack([ u, v], 3)
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

    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    out = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return out

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
