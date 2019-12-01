## For Orbit object
## for multiple mpis
## tring to do Learned gradient
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.compat.v1 import ConfigProto
from mpl_toolkits.mplot3d import Axes3D
import cv2

from view_gen import generateWebGL
from utils import *
from localpath import getLocalPath

#import memory_saving_gradients
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_collection
#from gen_tfrecord import findCameraSfm

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("invz", False, "using inverse depth, ignore dmax in this case")
tf.app.flags.DEFINE_boolean("restart", False, "restart training")
tf.app.flags.DEFINE_boolean("predict", False, "making a video")
tf.app.flags.DEFINE_boolean("endvdo", False, "making a last video frame")
tf.app.flags.DEFINE_float("scale", 0.75, "scale input image by")
tf.app.flags.DEFINE_integer("subscale", 8, "downscale factor for the sub layer")

tf.app.flags.DEFINE_integer("layers", 25, "number of planes")
tf.app.flags.DEFINE_integer("sublayers", 2, "number of sub planes")
tf.app.flags.DEFINE_integer("epoch", 1000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_integer("index", 0, "index number")

tf.app.flags.DEFINE_string("dataset", "temple0", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "cen0", "input tfrecord")

#tf.app.flags.DEFINE_string("ref_img", "0051.png", "reference image such that MPI is perfectly parallel to")
if FLAGS.dataset == "temple0":
    ref_img = ["0040.png", "0045.png", "0051.png", "0057.png", "0032.png", "0033.png", "0039.png", "0292.png", "0040.png"]
    ref_ID = ["354632085", "1221962097", "1004312245", "1902466051", "164864196", "949584407", "496808732", "228538494","354632085"]
elif FLAGS.dataset == "lib2":
    ref_img = ["2_00000","2_00004","2_00008","2_00012","2_00016","2_00000"]
elif FLAGS.dataset in ["acup11"]:
    ref_img = ["4_00000","4_00004","4_00008","4_00012","4_00016","4_00020","4_00024","4_00028","4_00032","4_00036","4_00000"]
elif FLAGS.dataset == "toro" :
    ref_img = ["2_00000","2_00004","2_00008","2_00012","2_00016","2_00020","2_00024","2_00028","2_00032","2_00036","2_00000"]
elif FLAGS.dataset in ["acup7", "acup6","toro2","dumdum"]:
    ref_img = ["0_00000","0_00004","0_00008","0_00012","0_00016","0_00020","0_00024","0_00028","0_00032","0_00036","0_00000"]
else:
    print("error: dataset not found")
    exit()
mpi_max = len(ref_img)-1


sub_sam = FLAGS.sublayers
num_mpi = FLAGS.layers
f = px = py = 0
ref_r = []
ref_t = []
ref_c = []
offset = 0
dmin, dmax = -1, -1
num_ = 2



def load_data(i,is_shuff=False):
  def parser(serialized_example):
    fs = tf.parse_single_example(
        serialized_example,
        features={
          "img": tf.FixedLenFeature([], tf.string),
          "r": tf.FixedLenFeature([9], tf.float32),
          "t": tf.FixedLenFeature([3], tf.float32),
          "h": tf.FixedLenFeature([], tf.int64),
          "w": tf.FixedLenFeature([], tf.int64),
        })

    fs["img"] = tf.to_float(tf.image.decode_png(fs["img"], 3)) / 255.0
    if FLAGS.scale < 1:
      fs["img"] = tf.image.resize(fs["img"], [h, w], tf.image.ResizeMethod.AREA)

    fs["r"] = tf.reshape(fs["r"], [3, 3])
    fs["t"] = tf.reshape(fs["t"], [3, 1])
    return fs


  localpp = "datasets/" + FLAGS.dataset + "/" + FLAGS.input + ".train"
  dataset = tf.data.TFRecordDataset([localpp])
  dataset = dataset.map(parser)
  if(is_shuff):  dataset = dataset.shuffle(5)
  dataset = dataset.repeat().batch(FLAGS.batch_size)

  return dataset.make_one_shot_iterator().get_next()

def setGlobalVariables():
  global f, px, py, ref_r, ref_t, ref_c, w, h

  path = findCameraSfm(FLAGS.dataset)
  with open(path, "r") as f:
    js = json.load(f)


  f = float(js["intrinsics"][0]["pxFocalLength"]) * FLAGS.scale
  px = float(js["intrinsics"][0]["principalPoint"][0]) * FLAGS.scale
  py = float(js["intrinsics"][0]["principalPoint"][1]) * FLAGS.scale
  w = int(int(js["intrinsics"][0]["width"]) * FLAGS.scale)
  h = int(int(js["intrinsics"][0]["height"]) * FLAGS.scale)

  st = 0
  for name in ref_img:
      for view in js["views"]:
        if name in view["path"]:
          for pose in js["poses"]:
            if pose["poseId"] == view["poseId"]:
              tt = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
              ref_r.append(tt.copy())
              tt = -ref_r[-1] * np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
              ref_t.append(tt.copy())
              tt = np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
              ref_c.append(tt.copy())
              st += 1
              break
          break

  global dmin, dmax, ccx, ccy, ccz,dx, dy, dz
  if dmin <0 and dmax<0:
     with open("datasets/" + FLAGS.dataset + "/planes.txt", "r") as fi:
       ccx,dx = [float(x) for x in fi.readline().split(" ")]
       ccy,dy = [float(x) for x in fi.readline().split(" ")]
       ccz,dz = [float(x) for x in fi.readline().split(" ")]

def computeHomography(r, t, d, ks=1,index=0):
  global f, px, py, ref_r, ref_t

  # equivalent to right multiplying [r; t] by the inverse of [ref_r, r ef_t]
  new_r = tf.matmul(r, tf.transpose(ref_r[index]))
  new_t = tf.matmul(tf.matmul(-r, tf.transpose(ref_r[index])), ref_t[index]) + t

  n = tf.constant([[0.0, 0.0, 1.0]])
  Ha = tf.transpose(new_r)

  Hb = tf.matmul(tf.matmul(tf.matmul(Ha, new_t), n), Ha)
  Hc = tf.matmul(tf.matmul(n, Ha), new_t)[0]

  k = tf.constant([[f, 0, px], [0, f, py], [0, 0, 1]])
  ki = tf.linalg.inv(k)

  if ks != 1:
    k = tf.constant([[f*ks, 0, px], [0, f*ks, py], [0, 0, 1]])

  return tf.matmul(tf.matmul(k, Ha + Hb / (-d-Hc)), ki)

def samplePlane(plane, r, t, d, ks=1,index=0):
  global w, h, offset
  nh = h + offset * 2
  nw = w + offset * 2

  H = computeHomography(r, t, d, ks,index)

  x, y = tf.meshgrid(list(range(w)), list(range(h)))
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)
  newCoords = tf.matmul(coords, tf.transpose(H))

  cx = tf.expand_dims(newCoords[:, :, 0] / newCoords[:, :, 2], 0)
  cy = tf.expand_dims(newCoords[:, :, 1] / newCoords[:, :, 2], 0)

  return bilinear_sampler(plane, cx + offset, cy + offset)

def getPlanes(index):
  dis=np.linalg.norm(np.transpose(ref_c[index]) - np.array([ccx,ccy,ccz]),2)
  cop = (dx+dy+dz)/3
  bias = -0.1
  dmin = dis-cop#+bias
  dmax = dis+cop+bias
  if FLAGS.invz:
    return 1/np.linspace(1, 0.1, num_mpi) * dmin
  else:
    return np.linspace(dmin, dmax, num_mpi)

def network(mpi, depth, bg, rot,tra, index, is_training):
  alpha = 1
  output = 0
  mask = 0.0
  imgs = []
  sublayers = []
  planes = getPlanes(index)
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])

  for i, v in enumerate(planes):
      plane = mpi[i]
      alphasi = depth[i]
      for j in range(sub_sam):
          vv = j/sub_sam
          dep = rplanes[i]*(1-vv) + rplanes[i+1]*(vv)
          depth_map = alphasi[:,:,j:j+1]
          img = samplePlane(tf.concat([plane, depth_map], -1),rot,tra, dep, 1,index)

          img = img[0]
          mul = img[:,:,3:4] * img[:,:,4:5]
          output += img[:,:,:3] * mul * alpha
          alpha *= (1-mul)
          depth_map = tf.image.resize(depth_map, [int(h/8), int(w/8)], tf.image.ResizeMethod.AREA)
          sublayers.append(depth_map)
          if j == 0:
              imgs.append(img)

  output += alpha*bg
  return output, imgs, sublayers

def network2(mpi, bg, rot,tra, index, is_training):
  #without sublayers
  alpha = 1
  output = 0
  imgs = []
  sublayers = []
  planes = getPlanes(index)
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])

  for i, v in enumerate(planes):
    dep = rplanes[i]
    plane = mpi[i]
    img = samplePlane(plane,rot,tra, dep, 1,index)
    img = img[0]
    mul = img[:,:,3:4]
    output += img[:,:,:3] * mul * alpha
    alpha *= (1-mul)
    imgs.append(img)

  output += alpha*bg
  return output, imgs, sublayers

def gen_depth(input,is_train,reuse=False):
    with tf.compat.v1.variable_scope('depth') as scope:
        if reuse: scope.reuse_variables()

        next = input#tf.expand_dims( input,0)
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            normalizer_fn=slim.batch_norm,
            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
            normalizer_params={"is_training": is_train}
            ):

            #next = slim.conv2d(next,16,[3,3],stride=2)
            for chanel in [32,48,32]:
                next = slim.conv2d(next,chanel,[3,3])
                next = slim.conv2d(next,chanel,[3,3], stride=2)

            next = slim.conv2d(next,FLAGS.sublayers,[3,3],
                    biases_initializer=tf.constant_initializer(-5.0),
                    activation_fn=tf.sigmoid)
    return next

def conv_grad0(grad,mpi,is_train,reuse=False):
    with tf.compat.v1.variable_scope('LGD') as scope:
        if reuse: scope.reuse_variables()
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            normalizer_fn=slim.batch_norm,
            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
            normalizer_params={"is_training": is_train}
            ):
            layers = []
            size = []
            a, b = h, w
            next = tf.concat([-0.1*grad,mpi],-1)
            for chanel in [32,64,72]:
                layers.append(next)
                size.append((a,b))
                a, b = round((a+0.1)/2), round((b+0.1)/2)
                next = slim.conv2d(next,chanel,[3,3])
                next = slim.conv2d(next,chanel,[3,3],stride=2)
            
            for i in range(3):
                next = next + slim.conv2d(next,72,[3,3])

            for i, lay in reversed(list(enumerate(layers))):
                next = tf.image.resize(next,size[i])
                next = tf.concat([next,lay],-1)
                next = slim.conv2d(next,24,[3,3])
            next = slim.conv2d(next,4,[3,3])
    return next

def conv2d(x,k,filter=3,stride=[1,1,1,1],name=''):
  w_int = tf.truncated_normal([filter, filter, x.shape[3].value, k], stddev=0.1)
  #w = tf.Variable(w_int,name=name+"_w")
  w = tf.get_variable(name+"_w",initializer=w_int, trainable=True)
  out = tf.nn.conv2d(x,w,strides=stride,padding='SAME')
  return lay_norm(out,name)

def lay_norm(x,name=''):
  a = tf.get_variable(name+"scale",shape=[x.shape[-1]],initializer=tf.initializers.random_normal())
  a = tf.reshape(tf.cast(a,x.dtype),[1,1,1,-1])
  b = tf.get_variable(name+"bias",shape=[x.shape[-1]],initializer=tf.initializers.zeros())
  b = tf.reshape(tf.cast(b,x.dtype),[1,1,1,-1])
  z = x - tf.reduce_mean(x,axis=[1,2],keepdims=True)
  #z *= tf.math.rsqrt(tf.reduce_mean(tf.square(z),axis=[1,2],keepdims=True))
  z = a * z + b
  return z

def lrelu(x):
   return tf.math.maximum(x,0.1*x)

def conv_grad1(grad,mpi,is_train,reuse=False):
    with tf.compat.v1.variable_scope('LGD') as scope:
        if reuse: scope.reuse_variables()
        layers = []
        size = []
        a, b = h, w
        next = tf.concat([grad,mpi],-1)
        #for chanel in [32,64,72]:
        #    layers.append(next)
        #    size.append((a,b))
        #    a, b = round((a+0.1)/2), round((b+0.1)/2)
        #    next = lrelu(lay_norm(conv2d(next,chanel,name="a"+str(chanel))))
        #    next = lrelu(lay_norm(conv2d(next,chanel,stride=[1,2,2,1],name="b"+str(chanel))))
            #next = lrelu(conv2d(next,chanel,name="a"+str(chanel)))
            #next = lrelu(conv2d(next,chanel,stride=[1,2,2,1],name="b"+str(chanel)))
        next = lay_norm(conv2d(next,24,name="begin"))
        for i in range(5):
            next = next + lrelu(lay_norm(conv2d(next,24,name="r"+str(i))))
            #next = next + lrelu(conv2d(next,72,name="r"+str(i)))
        #next = tf.Print(next,[next])

        #for i, lay in reversed(list(enumerate(layers))):
        #    next = tf.image.resize(next,size[i])
        #    next = tf.concat([next,lay],-1)
        #    next = lrelu(lay_norm(conv2d(next,32,name="c"+str(i))))
            #next = lrelu(conv2d(next,32,name="c"+str(i)))
        #next = grad+conv2d(next,4)
        next = grad+lay_norm(conv2d(next,4))
    return next

def conv_grad2(grad,mpi,is_train,reuse=False):
    with tf.compat.v1.variable_scope('LGD') as scope:
        if reuse: scope.reuse_variables()
        layers = []
        size = []
        a, b = h, w
        next = tf.concat([grad,tf.sigmoid(mpi)],-1)
        next = lrelu(conv2d(next,16,name="begin"))
        for i,chanel in enumerate([24,32,32,24]):
          next = lrelu((conv2d(next,chanel,name="a"+str(i))))
        #for i,chanel in enumerate([24,32,24]):
        #  next = lrelu((conv2d(next,chanel,filter=1,name="b"+str(i))))
        next = tf.concat([grad,next],-1)
        next = lrelu((conv2d(next,24,name="last1")))
        next = grad+conv2d(next,4,name="last2")
    return next

def ODE_Block(x,kernel,rate=[1, 1, 1, 1],block='ODE'):
    with tf.variable_scope(block):
        if tf.shape(x)[3] != kernel:
          x = (conv2d(x, kernel,filter=1, name='reshape'))
            #x = tf.layers.conv2d(x, kernel, [3, 3],padding="same", name="reshape")

        W = tf.Variable(tf.truncated_normal([3, 3, kernel, kernel], stddev=0.1), name="Weight")
        a = tf.get_variable("scale",shape=[x.shape[-1]],initializer=tf.initializers.random_normal())
        a = tf.reshape(tf.cast(a,x.dtype),[1,1,1,-1])
        b = tf.get_variable("bias",shape=[x.shape[-1]],initializer=tf.initializers.zeros())
        b = tf.reshape(tf.cast(b,x.dtype),[1,1,1,-1])

        @tf.custom_gradient  
        def ODE_func(x,w):
          def ODE_layer(x,t=0): #conv2, batch_norm, leaky_relu
            x = tf.nn.conv2d(x, w, [1,1,1,1],dilations=rate, padding='SAME')
            z = x - tf.reduce_mean(x,axis=[0,1,2],keepdims=True)
            z *= tf.math.rsqrt(tf.reduce_mean(tf.square(z)+1e-10,axis=[0,1,2],keepdims=True))
            #z = a * z + b
            out = z#tf.math.maximum(z,0.1*z)
            return out
            
          if (True): #RK4
              K1 = ODE_layer(x)
              K2 = ODE_layer(x + 0.5*K1)
              K3 = ODE_layer(x + 0.5*K2)
              K4 = ODE_layer(x + 1.0*K3)
              out = x + K1/6. + K2/3. + K3/3. + K4/6.
          else:   #RK2
              K1 = ODE_layer(x)
              K2 = ODE_layer(x + 1.0*K1)
              out = x + K1/2. + K2/2.

          def grad(dy):
            def aug_dyn(zab):
              z, a, b = zab[0], zab[1], zab[2]
              with tf.GradientTape(persistent=True) as t:
                t.watch((z,w))
                f_out = -ODE_layer(z)
              out_a, out_b = t.gradient(f_out,(z,w),output_gradients=-a)
              del t
              return (f_out, out_a, out_b)

            (z0, d_x, d_w) = aug_dyn((out,dy,tf.zeros_like(w)))
            #z0 += out
            d_x += dy
            return (d_x, d_w)
          return out, grad
    return ODE_func(x,W)

def conv_grad(grad,mpi,is_train,reuse=False):
    with tf.compat.v1.variable_scope('LGD') as scope:
        if reuse: scope.reuse_variables()
        next = tf.concat([grad,(mpi)],-1)
        next = grad+lrelu(conv2d(next,4,name="begin"))
        for i in range(3):
          next = ODE_Block(next,4,block="1ODE"+str(i))
        next = next + lrelu(conv2d(tf.concat([next,tf.sigmoid(mpi)],-1),4,name="mid"))
        for i in range(3):
          next = ODE_Block(next,4,block="2ODE"+str(i))
        next = next + lrelu(conv2d(tf.concat([next,grad],-1),4,name="mid2"))
        for i in range(3):
          next = ODE_Block(next,4,block="3ODE"+str(i))
        return next + conv2d(next,4,name="last0")

def train():
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    rot = tf.compat.v1.placeholder(tf.float32, shape=[3,3], name='rotation')
    tra = tf.compat.v1.placeholder(tf.float32, shape=[3,1], name='translation')
    real_img = tf.compat.v1.placeholder(tf.float32, shape=[h,w,3], name='ex_img')

    int_mpi1 = np.random.uniform(-1, 0,[num_mpi, h, w, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-1,[num_mpi, h, w, 1]).astype(np.float32)
    int_mpi = np.concatenate([int_mpi1,int_mpi2],-1)
    depth_init = np.random.uniform(-5,0,[num_mpi, int(h/ FLAGS.subscale), int(w/ FLAGS.subscale), sub_sam]).astype(np.float32)

    bg = tf.compat.v1.get_variable("Net_bg", initializer=np.array([0.8,0.8,0.8],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)

    lr1 = tf.compat.v1.train.exponential_decay(0.1,lod_in,1000,0.2)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr1)
    lr2 = tf.compat.v1.train.exponential_decay(0.0005,lod_in,1000,0.2)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(lr2)
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index%mpi_max)):
        mpi = tf.compat.v1.get_variable("mpi", initializer=int_mpi, trainable=True)
        mpi_sig = tf.sigmoid(mpi)
        img_out, shifts, sss = network2(mpi_sig, bg, rot,tra,(FLAGS.index)%mpi_max, True)
        long = tf.concat(shifts, 1)


    with tf.compat.v1.variable_scope("loss%d"%(FLAGS.index%mpi_max)):
        tva = tf.constant(0.1)
        tvc = tf.constant(0.005) 
        mpiColor = mpi_sig[:, :, :, :3]
        mpiAlpha = mpi_sig[:, :, :, 3:4]
        loss =  100000 * tf.reduce_mean(tf.square(img_out - real_img))
        loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
        loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
    grad_mpi = tf.gradients(loss,mpi)

    t_vars = tf.compat.v1.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'dis' not in var.name]

    with tf.compat.v1.variable_scope("post%d"%(FLAGS.index%mpi_max)):
        image_out = tf.clip_by_value(img_out,0.0,1.0)
        train_op = slim.learning.create_train_op(loss,optimizer,variables_to_train=g_vars)
        #train_dis = slim.learning.create_train_op(d_loss,optimizer,variables_to_train=d_vars)
        
        long = tf.clip_by_value(long,0.0,1.0)

        summary = tf.compat.v1.summary.merge([
                        tf.compat.v1.summary.scalar("all_loss", loss),
                        tf.compat.v1.summary.image("out",tf.expand_dims(image_out,0)),
                        tf.compat.v1.summary.image("origi",tf.expand_dims(real_img,0)),
                        tf.compat.v1.summary.image("alpha",tf.expand_dims(long[:,:,3:4],0)),
                        tf.compat.v1.summary.image("color",tf.expand_dims(long[:,:,:3]*long[:,:,3:4],0)),
                        ])

    new_loss = 0
    in_grad = tf.compat.v1.placeholder(tf.float32, shape=[num_mpi,h,w,4], name='input_grad')
    old_mpi = tf.compat.v1.placeholder(tf.float32, shape=[num_mpi,h,w,4], name='old_mpi')
    new_grad = conv_grad(in_grad,old_mpi,True)
    new_mpi = tf.sigmoid(old_mpi-lr1*new_grad)
    #new_loss +=  10000 * tf.reduce_mean(tf.square(tf.stop_gradient(mpi_sig)-new_mpi))
    #new_loss2 =  1000 * tf.reduce_mean(tf.square(in_grad - new_grad))
    new_img, _, __ = network2(new_mpi, tf.stop_gradient(bg), rot,tra,(FLAGS.index)%mpi_max, True)
    new_loss +=  100000 * tf.reduce_mean(tf.square(real_img - new_img))
    new_loss += tvc * tf.reduce_mean(tf.image.total_variation (new_mpi[:, :, :, :3]))
    new_loss += tva * tf.reduce_mean(tf.image.total_variation (new_mpi[:, :, :, 3:4]))
    sta_grad = conv_grad(old_mpi*0,old_mpi,True,reuse=True)
    sta_loss = tf.reduce_mean(tf.square(sta_grad))
    t_vars = tf.compat.v1.trainable_variables()
    l_vars = [var for var in t_vars if 'LGD' in var.name]
    m_vars = [var for var in t_vars if 'mpi' in var.name]
    optimizer3 = tf.compat.v1.train.GradientDescentOptimizer(lr1)
    train_op2 = slim.learning.create_train_op(new_loss,optimizer2,variables_to_train=l_vars)
    train_op3 = slim.learning.create_train_op(sta_loss,optimizer2,variables_to_train=l_vars)
    grads_and_vars = zip([new_grad], m_vars)
    #app_grad = [v.assign(v-lr1*in_grad) for v in tf.contrib.framework.get_variables_by_name("mpi")]
    app_grad = optimizer3.apply_gradients(grads_and_vars)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    localpp = "TensorB/"+FLAGS.dataset
    if FLAGS.index==0:
      if os.path.exists(localpp):
        os.system("rm -rf " +localpp )
    if not os.path.exists(localpp):
      os.makedirs(localpp)
    writer = tf.compat.v1.summary.FileWriter(localpp)
    #writer.add_graph(sess.graph)

    saver = tf.train.Saver()
    #localpp = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str(FLAGS.index%mpi_max))
    localpp = './model/' + FLAGS.dataset +'/tem'+str(FLAGS.index%mpi_max)
    if not os.path.exists(localpp):
        os.makedirs(localpp)

    features = load_data(0,is_shuff = True)
    sess = tf.compat.v1.Session(config=config)
    if not FLAGS.restart:
        sess.run(tf.compat.v1.global_variables_initializer())
        t_vars = slim.get_variables_to_restore()
        vars_to_restore = [var for var in t_vars if 'LGD' in var.name ]
        print(vars_to_restore)
        saver = tf.train.Saver(vars_to_restore)
        ckpt = tf.train.latest_checkpoint(localpp )
        saver.restore(sess, ckpt)
        #saver = tf.train.Saver()

    elif FLAGS.index>mpi_max-1:
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        ckpt = tf.train.latest_checkpoint(localpp )
        saver.restore(sess, ckpt)
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

        if FLAGS.index>0:
          t_vars = slim.get_variables_to_restore()
          localppp = './model/' + FLAGS.dataset +'/tem'+str((FLAGS.index-1)%mpi_max)
          vars_to_restore = [var for var in t_vars if 'depth' in var.name and 'post' not in var.name]
          #vars_to_restore = vars_to_restore[:5]
          #print(len(vars_to_restore),vars_to_restore)
          saver = tf.train.Saver(vars_to_restore)
          ckpt = tf.train.latest_checkpoint(localppp )
          saver.restore(sess, ckpt)
          saver = tf.train.Saver()
    los = 0
    for i in range(FLAGS.epoch + 3):
        feed = sess.run(features)
        lodin = int(FLAGS.index/mpi_max)*int(FLAGS.epoch/2)
        #_,los = sess.run([train_op,loss],feed_dict={lod_in:i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
        feedlist = {lod_in:i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]}
        gg,ompi = sess.run([grad_mpi,mpi],feed_dict=feedlist)
        guid_lim = 2000 if FLAGS.restart else 200
        if i<guid_lim and i%10!=9:
            _, los0 = sess.run([train_op,loss],feed_dict=feedlist)
            feedlist = {lod_in:i,in_grad:gg[0],old_mpi:ompi,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]}
            _,gg2,los = sess.run([train_op2,new_grad,new_loss],feed_dict=feedlist)
        else:
            feedlist = {lod_in:i,in_grad:gg[0],old_mpi:ompi,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]}
            _,gg2,ompi,los = sess.run([train_op2,new_grad,mpi,new_loss],feed_dict=feedlist)
            sess.run(app_grad,feed_dict={lod_in:i,in_grad:gg2,old_mpi:ompi})
        
        

        #for j in range(5):
        #    _,dlos = sess.run([train_dis,d_loss],feed_dict={lod_in:600,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
    
        if i%100==0:
            print(FLAGS.index,i, "loss = " ,los0,los)
            #print("grad",np.ndarray.max(gg[0]),np.ndarray.min(gg[0]))
            #print("New grad",np.ndarray.max(gg2),np.ndarray.min(gg2))
        if i%20 == 0:
            if FLAGS.restart: sess.run(train_op3,feed_dict=feedlist)
            summ = sess.run(summary,feed_dict={lod_in:i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
            writer.add_summary(summ,i)
        if i%200==1:
            saver.save(sess, localpp + '/' + str(i))


def predict():
    global w, h, offset, f
    def parser(serialized_example):
      fs = tf.parse_single_example(
          serialized_example,
          features={
            "r": tf.FixedLenFeature([9], tf.float32),
            "t": tf.FixedLenFeature([3], tf.float32),
          })
      fs["r"] = tf.reshape(fs["r"], [3, 3])
      fs["t"] = tf.reshape(fs["t"], [3, 1])
      return fs

    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    rot = tf.compat.v1.placeholder(tf.float32, shape=[3,3], name='rotation')
    tra = tf.compat.v1.placeholder(tf.float32, shape=[3,1], name='translation')


    testset = tf.data.TFRecordDataset(["datasets/" + FLAGS.dataset + "/" + FLAGS.input +".test"])
    testset = testset.map(parser).repeat().batch(1).make_one_shot_iterator()
    features = testset.get_next()


    bg = tf.compat.v1.get_variable("Net_bg", initializer=np.array([1,1,1],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)

    depth_init = np.random.uniform(-5,0,[num_mpi, int(h/ FLAGS.subscale), int(w/ FLAGS.subscale), sub_sam]).astype(np.float32)

    mpi = np.zeros([num_mpi, h, w, 4],dtype=np.float32)
    mpi[0] = [1.,0.,0.,.95]
    mpi[1] = [1.,.5,0.,.95]
    mpi[2] = [1.,1.,0.,.95]
    mpi[3] = [.5,1.,0.,.95]

    depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
    depth = tf.sigmoid(depth)
    #depth = tf.image.resize(depth, [h, w], align_corners=True)


    with tf.compat.v1.variable_scope("Net"):
        mpi = tf.compat.v1.get_variable("mpi", initializer=mpi, trainable=False)
        mpi = tf.sigmoid(mpi)
        #depth = gen_depth(mpi,False)
        depth = tf.image.resize(depth, [h, w], align_corners=True)
        img_out, shifts, sss = network(mpi, depth, bg, rot,tra,FLAGS.index%mpi_max, False)
        long = tf.concat(shifts, 1)

    with tf.compat.v1.variable_scope("post"):
        image_out= tf.clip_by_value(img_out,0.0,1.0)
        longs = tf.clip_by_value(long,0.0,1.0) 

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    #localpp = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index)%mpi_max))
    localpp = './model/' + FLAGS.dataset +'/tem'+str((FLAGS.index)%mpi_max)
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)


    if True:
        for i in range(0,300,1):
          try:
            out0 = cv2.imread("result/frame/"+FLAGS.dataset+"_%04d.png"%((300*(FLAGS.index%mpi_max) + i-300)%(300*mpi_max)))
            out0 = cv2.cvtColor(out0, cv2.COLOR_BGR2RGB)/255.0
            feed = sess.run(features)
            if(i%50==0): print(i, ((300*(FLAGS.index%mpi_max) + i-300)%(300*mpi_max)))
            out = sess.run(image_out,feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
            ii = i/300
            out =  np.clip(out*ii+out0*(1-ii),0.0,1.0)
            plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%((300*(FLAGS.index%mpi_max) + i-300)%(300*mpi_max)),out)
          except:
            feed = sess.run(features)
            if(i%50==0): print(i, ((300*(FLAGS.index%mpi_max) + i-300)%(300*mpi_max)))
            out = sess.run(image_out,feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
            plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%((300*(FLAGS.index%mpi_max) + i-300)%(300*mpi_max)),out)

        if(not FLAGS.endvdo):
          for i in range(0,300,1):
              feed = sess.run(features)
              if(i%50==0): print(i, (300*(FLAGS.index%mpi_max) + i))
              out = sess.run(image_out,feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
              plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%(300*(FLAGS.index%mpi_max) + i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p result/'+FLAGS.dataset+'_moving.mp4'
        print(cmd)
        os.system(cmd)

    if True:
      webpath = "webpath/"  #"/var/www/html/orbiter/"
      if not os.path.exists(webpath + FLAGS.dataset):
          os.system("mkdir " + webpath + FLAGS.dataset)


      for ii in range(1):
        ref_rt = np.array(ref_r[FLAGS.index%mpi_max+ii:FLAGS.index%mpi_max+ii+1])
        ref_tt = np.array(ref_t[FLAGS.index%mpi_max+ii:FLAGS.index%mpi_max+ii+1])
        print(ref_rt.shape)
        ret, sublay = sess.run([mpi,sss],feed_dict={features['r']:ref_rt,features['t']:ref_tt})
        sublayers = []
        mpis = []
        sublayers_combined = []
        print("sublay",sublay[0].shape,len(sublay))
        for i in range(num_mpi):
            mpis.append(ret[i,:,:,:4])
            ls = []
            for j in range(sub_sam):
                ls.append(sublay[sub_sam*i+j])
            ls = np.concatenate(ls,0)
            #ls = np.expand_dims(ls,-1)
            ls = np.tile(ls,(1,1,3))
            sublayers.append(ls)
            ls = np.reshape(ls, (sub_sam,sublay[0].shape[0],sublay[0].shape[1],3))
            ls = np.clip(np.sum(ls, 0), 0.0, 1.0)
            sublayers_combined.append(ls)
            #out = np.rot90(out,1)

        mpis = np.concatenate(mpis, 1)
        plt.imsave(webpath + FLAGS.dataset+ "/mpi%02d.png"%(FLAGS.index%mpi_max+ii), mpis)
        #plt.imsave(webpath + FLAGS.dataset+ "/mpi_alpha"+str(ii)+".png", np.tile(mpis[:, :, 3:], (1, 1, 3)))
        sublayers = np.concatenate(sublayers, 1)
        sublayers = np.clip(sublayers, 0, 1)
        plt.imsave(webpath + FLAGS.dataset+ "/sublayer%02d.png"%(FLAGS.index%mpi_max+ii), sublayers)
        #sublayers_combined = np.concatenate(sublayers_combined, 1)
        #plt.imsave(webpath + FLAGS.dataset+ "/sublayers_combined.png", sublayers_combined)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi.png", mpis)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi_alpha.png", np.tile(mpis[:, :, 3:], (1, 1, 3)))

      namelist = "["
      for ii in range(FLAGS.index%mpi_max+1):
        namelist += "\"%02d\","%(ii)
      namelist += "]"
      print(namelist)

      with open(webpath + FLAGS.dataset+ "/extrinsics%02d.txt"%(FLAGS.index%mpi_max), "w") as fo:
        for i in range(3):
          for j in range(3):
            fo.write(str(ref_r[FLAGS.index%mpi_max][ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t[(FLAGS.index%mpi_max)])]) + "\n")

      generateWebGL(webpath + FLAGS.dataset+ "/index.html", w, h, getPlanes(FLAGS.index%mpi_max),namelist,sub_sam, f, px, py)

def main(argv):
    setGlobalVariables()
    if FLAGS.predict:
        predict()
    else:
        train()
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()
