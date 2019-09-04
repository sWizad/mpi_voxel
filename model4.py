## fixed render box
## 2 times lookup (cubic depth and hi res mpi)
## for multiple mpis
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.compat.v1 import ConfigProto
from mpl_toolkits.mplot3d import Axes3D
import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
tf.__dict__["gradients"] = memory_saving_gradients.gradients_collection

from utils import *
#from gen_tfrecord import findCameraSfm

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("invz", False, "using inverse depth, ignore dmax in this case")
tf.app.flags.DEFINE_boolean("predict", False, "make a video")
tf.app.flags.DEFINE_float("scale", 1.0, "scale input image by")

tf.app.flags.DEFINE_integer("layers", 25, "number of planes")
tf.app.flags.DEFINE_integer("epoch", 500, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_string("dataset", "temple1", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "temple-all", "input tfrecord")

tf.app.flags.DEFINE_string("ref_img", "0004.png", "reference image such that MPI is perfectly parallel to")


laten_h, laten_w, laten_d, laten_ch = 60,60,60,4#50, 50, 50, 4
#laten_h, laten_w, laten_d, laten_ch = 35,20,20,4
#texture resolution
size_h, size_w = 271, 362#362, 483
sub_sam = 20
num_mpi = 6
f = px = py = 0
ref_r = ref_t = 0
offset = 0
dmin, dmax = -1, -1

def load_data():
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
      fs["img"] = tf.image.resize_images(fs["img"], [h, w], tf.image.ResizeMethod.AREA)

    fs["r"] = tf.reshape(fs["r"], [3, 3])
    fs["t"] = tf.reshape(fs["t"], [3, 1])
    return fs

  # np.random.shuffle(filenames)
  dataset = tf.data.TFRecordDataset(["datasets/" + FLAGS.dataset + "/" + FLAGS.input + ".train"])
  dataset = dataset.map(parser)
  #dataset = dataset.shuffle(5)
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
  print("h =",h," w =",w)


  st = 0
  for view in js["views"]:
    if FLAGS.ref_img in view["path"]:
      for pose in js["poses"]:
        if pose["poseId"] == view["poseId"]:
          ref_r = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
          ref_t = -ref_r * np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
          ref_c = np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
          st = 1
          break
      break

  print(FLAGS.ref_img)
  print(view["path"])



  global dmin, dmax, ccx, ccy, ccz,dx, dy, dz
  if dmin <0 and dmax<0:
     with open("datasets/" + FLAGS.dataset + "/planes.txt", "r") as fi:
       ccx,dx = [float(x) for x in fi.readline().split(" ")]
       ccy,dy = [float(x) for x in fi.readline().split(" ")]
       ccz,dz = [float(x) for x in fi.readline().split(" ")]
     #dx *= 1.1
     #dy *= 1.1
     #dz *= 1.3
     #ccx -= 0.15
     #ccy -= 0.15
     #ccz -= 0.35
     ccx -= 0.5#1.1
     ccy -= 1.1#1.1
     ccz -= 0.25#0.25
     dis=np.linalg.norm(np.transpose(ref_c) - [ccx,ccy,ccz],2)
     #cop = (dx+dy+dz+2*np.amax([dx,dy,dz]))/5
     #cop = (dx+dy+3*dz)/6.5
     cop = (dx+dy+3*dz)/5
     dmin = dis-cop
     dmax = dis+cop

#def get_pixel_value(mpi, x, y, z):#
#    indices = tf.stack([z, x, y], -1)
#    return tf.gather_nd(mpi, indices)

def get_pixel_value(mpi, y, x, z):
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

  Ia = get_pixel_value(mpi, y0, x0, z0)
  Ib = get_pixel_value(mpi, y0, x1, z0)
  Ic = get_pixel_value(mpi, y1, x0, z0)
  Id = get_pixel_value(mpi, y1, x1, z0)


  Ia1 = get_pixel_value(mpi, y0, x0, z1)
  Ib1 = get_pixel_value(mpi, y0, x1, z1)
  Ic1 = get_pixel_value(mpi, y1, x0, z1)
  Id1 = get_pixel_value(mpi, y1, x1, z1)

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

  Ia = get_pixel_value(mpi, y0, x0, zz)
  Ib = get_pixel_value(mpi, y0, x1, zz)
  Ic = get_pixel_value(mpi, y1, x0, zz)
  Id = get_pixel_value(mpi, y1, x1, zz)

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

def computeHomography(r, t, d, ks=1):
  global f, px, py, ref_r, ref_t

  # equivalent to right multiplying [r; t] by the inverse of [ref_r, r ef_t]
  new_r = tf.matmul(r, tf.transpose(ref_r))
  new_t = tf.matmul(tf.matmul(-r, tf.transpose(ref_r)), ref_t) + t

  n = tf.constant([[0.0, 0.0, 1.0]])
  Ha = tf.transpose(new_r)

  Hb = tf.matmul(tf.matmul(tf.matmul(Ha, new_t), n), Ha)
  Hc = tf.matmul(tf.matmul(n, Ha), new_t)[0]

  k = tf.constant([[f, 0, px], [0, f, py], [0, 0, 1]])
  ki = tf.linalg.inv(k)

  if ks != 1:
    k = tf.constant([[f*ks, 0, px], [0, f*ks, py], [0, 0, 1]])

  return tf.matmul(tf.matmul(k, Ha + Hb / (-d-Hc)), ki)

def samplePlane(plane, r, t, d, ks=1):
  global w, h, offset
  nh = h + offset * 2
  nw = w + offset * 2

  H = computeHomography(r, t, d, ks)

  x, y = tf.meshgrid(list(range(w)), list(range(h)))
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)
  newCoords = tf.matmul(coords, tf.transpose(H))

  cx = tf.expand_dims(newCoords[:, :, 0] / newCoords[:, :, 2], 0)
  cy = tf.expand_dims(newCoords[:, :, 1] / newCoords[:, :, 2], 0)

  return bilinear_sampler(plane, (cx + offset)*size_w/w, (cy + offset)*size_h/h)

def sampleDepth(latent,d):
    global f, px, py, ref_r, ref_t, size_w, size_h
    x, y = np.meshgrid(np.linspace(-1,1,size_w), np.linspace(-1,1,size_h))
    wan = np.ones_like(x)

    coords = np.stack([x*d, y*d, wan*d, wan], -1).reshape(size_h,size_w,4,1) #(h,w,4)
    k = [[2*f/size_w, 0      , 2*px/size_w-1  , 0],
         [0    , -2*f/size_h , -2*py/size_h+1 , 0],
         [0    , 0      , 1         , 0],
         [0    , 0      , 0         , 1]]
    ki = np.linalg.inv(k)
    mv = np.zeros([4,4])
    mv[:3,:3] = ref_r
    for i in range(3): mv[i,3] = ref_t[i]
    mv[3,3] =1
    mvi= np.linalg.inv(mv)
    ncoords = np.matmul(mvi,np.matmul(ki,coords))[:,:,:,0]
    #print(d,ncoords)
    #exit()
    #ncoords = tf.convert_to_tensor(ncoords,dtype=tf.float32)
    cx = (ncoords[:, :, 0]-ccx+dx)/dx/2
    cy = (ncoords[:, :, 1]-ccy+dy)/dy/2
    cz = (ncoords[:, :, 2]-ccz+dz)/dz/2
    #upb = tf.constant(1.1,dtype=tf.float32)
    #p1 = nearest3d_sampler(latent, cx, cy, cz)
    p1 = linear3d_sampler(latent, cx, cy, cz)
    tf.add_to_collection("checkpoints", p1)
    b =tf.cast(tf.greater(cx,-0.1),tf.float32)*tf.cast(tf.greater(cy,-0.1),tf.float32)*tf.cast(tf.greater(cz,-0.1),tf.float32)
    b *= tf.cast(tf.greater(1.1,cx),tf.float32)*tf.cast(tf.greater(1.1,cy),tf.float32)*tf.cast(tf.greater(1.1,cz),tf.float32)
    return p1*tf.expand_dims(b,-1)

def testplot( a=0):
    k = [[2*f/w, 0      , 2*px/w-1  , 0],
         [0    , -2*f/h , -2*py/h+1 , 0],
         [0    , 0      , 1         , 0],
         [0    , 0      , 0         , 1]]
    ki = np.linalg.inv(k)
    mv = np.zeros([4,4])
    mv[:3,:3] = ref_r
    for i in range(3): mv[i,3] = ref_t[i]
    mv[3,3] =1
    mvi= np.linalg.inv(mv)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def matcoor(d,color):
        print(d)
        x, y = np.meshgrid(np.linspace(-1,1,2), np.linspace(-1,1,2))
        wan = np.ones_like(x)
        coords = np.stack([x*d, y*d, wan*d, wan], -1).reshape(2,2,4,1) #(h,w,4)
        ncoords = np.matmul(mvi,np.matmul(ki,coords))[:,:,:,0]
        ax.plot_surface(ncoords[:,:,0], ncoords[:,:,1], ncoords[:,:,2],cmap=color,linewidth=0, antialiased=False)
    matcoor(dmin,cm.winter)
    matcoor((dmin+dmax)/2,cm.spring)
    matcoor(dmax,cm.summer)

    def randrange(n, vmin, vmax):
        return (vmax - vmin)*np.random.rand(n) + vmin
    xs = randrange(500, ccx-dx, ccx+dx)
    ys = randrange(500, ccy-dy, ccy+dy)
    zs = randrange(500, ccz-dz, ccz+dz)
    ax.scatter(xs, ys, zs, c='r')

    ax.scatter(ref_c[0], ref_c[1], ref_c[2], c='g')
    plt.show()
    exit()


def getPlanes():
  if FLAGS.invz:
    return 1/np.linspace(1, 0.0001, num_mpi) * dmin
  else:
    return np.linspace(dmin, dmax, num_mpi)

def network(mpi, latent, bg, features, is_training):
  alpha = 1
  output = 0.0
  mask = 0.0
  imgs = []
  planes = getPlanes()
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])

  for i, v in enumerate(planes):
      aa = 1
      out = 0
      for j in range(sub_sam):
          vv = j/sub_sam
          dep = rplanes[i]*(1-vv) + rplanes[i+1]*(vv)
          depth = sampleDepth(latent,dep)
          img = samplePlane(tf.concat([mpi[i], depth], -1),features["r"][0],features["t"][0], dep, 1)
          tf.add_to_collection("checkpoints", img)
          img = img[0]
          out += img[:,:,:4]*img[:,:,4:5]*aa
          aa  *= (1-img[:,:,4:5])
          if j == 1:
              imgs.append(img)
      output += out[:,:,:3]*out[:,:,3:4]*alpha
      mask += out[:,:,3:4]*alpha
      alpha *= (1-out[:,:,3:4])
  output += (1-mask)*bg
  return output, imgs

def train():
    lod_in = tf.placeholder(tf.float32, shape=[], name='lod_in')
    features = load_data()


    latent = np.random.uniform(-5,-0.1,[laten_d, laten_h, laten_w, 1]).astype(np.float32)
    #latent = np.ones([laten_d, laten_h, laten_w, 1],dtype=np.float32)
    mpi = np.random.uniform(-3,1,[num_mpi, size_h, size_w, 4]).astype(np.float32)

    bg = tf.get_variable("bg", initializer=np.array([1,1,1],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)
    latent = tf.get_variable("depth", initializer=latent, trainable=True)
    latent = tf.sigmoid(latent)
    #latent = tf.concat([tf.sigmoid(latent[:,:,:,:3]), tf.sigmoid(latent[:,:,:,3:]*3)],-1)
    mpi = tf.get_variable("mpi", initializer=mpi, trainable=True)
    noise = 2*tf.random_uniform(mpi.shape)-1
    mpi = tf.sigmoid(mpi+noise*0.0)
    #mpia = tf.where(tf.random_uniform(mpi.shape) - mpip < 0, tf.ones_like(mpi), tf.zeros_like(mpi))
    #mpi = (mpip + mpia*3)/4


    img_out, shifts = network(mpi, latent, bg, features, False)
    long = tf.concat(shifts, 1)

    with tf.compat.v1.variable_scope("loss"):
        mpiColor = mpi[:, :, :, :3]
        mpiAlpha = mpi[:, :, :, 3:4]
        mask = tf.reduce_mean(features["img"][0],-1,keepdims=True)
        mask = tf.cast(tf.greater(mask,0.015),tf.float32)
        loss =  100000 * tf.reduce_mean(mask*tf.square(img_out - features["img"][0]))
        #loss += 0.05 * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
        loss += 0.001 * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
        #loss += 0.0005 * tf.reduce_mean(tf.image.total_variation(mpiColor))
        #loss += 0.0001 * tf.reduce_mean(tf.image.total_variation(mpiColor))

    #long = tf.concat(shifts, 1)
    img_out = tf.clip_by_value(img_out,0.0,1.0)
    long = tf.clip_by_value(long,0.0,1.0)

    lr = 0.1
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = slim.learning.create_train_op(loss,optimizer)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if os.path.exists("TensorB/"+FLAGS.dataset):
        os.system("rm -rf TensorB/" + FLAGS.dataset)
    os.makedirs("TensorB/"+FLAGS.dataset)
    writer = tf.compat.v1.summary.FileWriter("TensorB/"+FLAGS.dataset)
    writer.add_graph(sess.graph)
    summary = tf.compat.v1.summary.merge([
                    tf.compat.v1.summary.scalar("all_loss", loss),
                    tf.compat.v1.summary.image("out",tf.expand_dims(img_out,0)),
                    tf.compat.v1.summary.image("alpha",tf.expand_dims(long[:,:,3:4],0)),
                    tf.compat.v1.summary.image("color",tf.expand_dims(long[:,:,:3],0)),
                    tf.compat.v1.summary.image("depth",tf.expand_dims(long[:,:,4:5],0)),
                    ])

    saver = tf.train.Saver()
    if not os.path.exists('./model/' + FLAGS.dataset +'/'+ FLAGS.input):
        os.makedirs('./model/' + FLAGS.dataset +'/'+ FLAGS.input)
    for i in range(FLAGS.epoch+3):
        _,los = sess.run([train_op,loss])
        if i%100==0:
            print(i, "loss = " ,los)
        if i%20 == 0:
           summ = sess.run(summary)
           writer.add_summary(summ,i)
        if i%200==0:
           saver.save(sess, './model/' + FLAGS.dataset +'/'+ FLAGS.input + '/' + str(i))


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

    lod_in = tf.placeholder(tf.float32, shape=[], name='lod_in')


    testset = tf.data.TFRecordDataset(["datasets/" + FLAGS.dataset + "/" + FLAGS.input + ".test"])
    testset = testset.map(parser).repeat().batch(1).make_one_shot_iterator()
    features = testset.get_next()


    latent = np.zeros([laten_d, laten_h, laten_w, 1],dtype=np.float32)

    latent[-2:laten_d,:,:,0] = 0.5
    for i in range(laten_d):
        ch, cw = int(laten_h/2-i/2), int(laten_w/2-i/4)
        latent[i,ch-10:ch+10,cw-5:cw+5,0] = 1.0

    mpi = np.zeros([num_mpi,  size_h, size_w, 4],dtype=np.float32)
    mpi[0] = [1.,0.,0.,.95]
    mpi[1] = [1.,.5,0.,.95]
    mpi[2] = [1.,1.,0.,.95]
    mpi[3] = [.5,1.,0.,.95]

    bg = tf.get_variable("bg", initializer=np.array([5,5,5],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)
    latent = tf.get_variable("depth", initializer=latent, trainable=False)
    latent = tf.sigmoid(latent)
    mpi = tf.get_variable("mpi", initializer=mpi, trainable=True)
    mpi = tf.sigmoid(mpi)


    img_out, shifts = network(mpi, latent, bg, features, False)

    long = tf.concat(shifts, 1)
    img_out = tf.clip_by_value(img_out,0.0,1.0)
    long = tf.clip_by_value(long,0.0,1.0)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.latest_checkpoint('./model/' + FLAGS.dataset +'/'+ FLAGS.input )
    saver.restore(sess, ckpt)

    if False:
        #bug = sess.run(long)
        #print(bug.shape)
        #out, bug = sess.run([img_out,latent])
        #print(bug.shape)
        plt.imshow(bug[:,:,:3])
        tt = np.concatenate([bug[1,:,:,0],bug[20,:,:,0],bug[40,:,:,0],bug[59,:,:,0]],1)
        plt.matshow(tt,cmap='gray')
        #plt.imshow(out)
        plt.show()
    else:
        #ff = sess.run(features)
        #print(fff)
        if os.path.exists("result/frame"):
            os.system("rm -rf result/frame")
        os.makedirs("result/frame")
        for i in range(200):
            print(i)
            out = sess.run(img_out)
            #out, bug = sess.run([img_out,long])
            #out = np.rot90(out,1)
            #plt.imsave("result/%04d.png"%(i),bug[:,:,:3])
            plt.imsave("result/frame/%04d.png"%(i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/\%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p result/' +FLAGS.input+ '.mp4'
        print(cmd)
        os.system(cmd)

    #out, bug = sess.run([img_out,debug])
    #plt.imshow(out)
    #plt.show()



def main(argv):
    #ss = load_data()
    #with tf.Session() as sess:
    #    print(sess.run(ss))
    setGlobalVariables()
    #testplot()
    #sampleDepth(0,dmin)
    if FLAGS.predict:
        predict()
    else:
        train()
    #predict()
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()
