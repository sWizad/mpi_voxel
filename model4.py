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
#import memory_saving_gradients
from view_gen import generateWebGL
import cv2

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_collection

from utils import *

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("invz", False, "using inverse depth, ignore dmax in this case")
tf.app.flags.DEFINE_boolean("predict", False, "make a video")
tf.app.flags.DEFINE_float("scale", 1.0, "scale input image by")

tf.app.flags.DEFINE_integer("layers", 25, "number of planes")
tf.app.flags.DEFINE_integer("epoch", 100, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_string("dataset", "temple1", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "temple-all", "input tfrecord")
tf.app.flags.DEFINE_string("version", "", "additional name")

tf.app.flags.DEFINE_string("ref_img", "0004.png", "reference image such that MPI is perfectly parallel to")


laten_h, laten_w, laten_d, laten_ch = 60,60,80,4#50, 50, 50, 4
#laten_h, laten_w, laten_d, laten_ch = 35,20,20,4
#texture resolution
size_h, size_w =  271, 362 # 362, 483 #
sub_sam = 15
num_mpi = 5
reuse_fac = 1
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
          image_path = "datasets/" + FLAGS.dataset + "/undistorted/"+view["path"].split("/")[-1][:-3]+"png"
          ref_photo = (plt.imread(image_path) * 255).astype(int)
          #ref_photo = cv2.resize(ref_photo,(h,w))
          #plt.imshow(ref_img)
          #plt.show()
          #exit()
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
     dx *= 1.0
     dy *= 1.0
     dz *= 1.0
     ccx -= 0.65
     ccy -= 0.15
     ccz -= 0.25 #.35

     ##set bball
     #dz *= 0.9
     #ccx -= 2.5
     #ccy -= 3.1#1.1
     #ccz -= 3.1#0.25

     dis = np.linalg.norm(np.transpose(ref_c) - [ccx,ccy,ccz],2)
     #cop = (dx+dy+dz+2*np.amax([dx,dy,dz]))/5
     #cop = (dx+dy+3*dz)/6.5
     cop = (dx+dy+3*dz)/8.0
     dmin = dis-cop
     dmax = dis+cop

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

def samplePlane(plane, r, t, d,theta, ks=1):
  global w, h, offset
  nh = h + offset * 2
  nw = w + offset * 2

  H = computeHomography(r, t, d, ks)

  x, y = tf.meshgrid(list(range(w)), list(range(h)))
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)

  coords = tf.stack([x, y, tf.ones_like(x)], 2)
  mapping = tf.constant([[2/w, 0.0, -1.0],[0.0, 2/h, -1.0],[0.0,0.0,1.0]])
  #with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
    #  sx = tf.get_variable("sx", initializer=2.0, trainable=True)
     # sy = tf.get_variable("sy", initializer=1.1, trainable=True)
      #theta = tf.stack([tf.stack([sx,0.0,0.0]),tf.stack([0.0,sy,0.0]),tf.stack([0.0,0.0,1.0])])
  #theta = tf.constant([[1.5,0,0],[0,1.0,0],[0,0,1]])
  #thetap = tf.matmul(tf.linalg.inv(mapping),tf.matmul(theta,mapping))
  #coords = tf.matmul(coords,tf.transpose(thetap))
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

    cx = (ncoords[:, :, 0]-ccx+dx)/dx/2
    cy = (ncoords[:, :, 1]-ccy+dy)/dy/2
    cz = (ncoords[:, :, 2]-ccz+dz)/dz/2

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
    return 1/np.linspace(1, 0.0001, num_mpi*reuse_fac) * dmin
  else:
    return np.linspace(dmin, dmax, num_mpi*reuse_fac)

def network(mpi, latent, bg, features, is_training):
  alpha = 1
  output = 0.0
  mask = 0.0
  imgs = []
  sublayers = []
  planes = getPlanes()
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])
  theta1 = tf.constant([[1.0,0,0],[0,1.0,0],[0,0,1]])
  theta2 = tf.constant([[2.0,0,0],[0,1.0,0],[0,0,1]])

  for i, v in enumerate(planes):
      aa = 1
      out = 0
      ls = []
      for j in range(sub_sam):
          vv = j/sub_sam
          dep = rplanes[i]*(1-vv) + rplanes[i+1]*(vv)
          depth = sampleDepth(latent,dep)
          #dd  = samplePlane(depth,features["r"][0],features["t"][0], dep, theta1, 1)
          #img = samplePlane(mpi[i],features["r"][0],features["t"][0], dep, theta2, 1)
          #img = tf.concat([img,dd],-1)
          img = samplePlane(tf.concat([mpi[i], depth], -1),features["r"][0],features["t"][0], dep, 1)
          tf.add_to_collection("checkpoints", img)
          img = img[0]
          out += img[:,:,:4]*img[:,:,4:5]*aa
          aa  *= (1-img[:,:,4:5])
          depth = tf.image.resize_images(depth, [int(h/8), int(w/8)], tf.image.ResizeMethod.AREA)
          sublayers.append(depth)
          if j == 1:
              imgs.append(img)
      output += out[:,:,:3]*out[:,:,3:4]*alpha
      mask += out[:,:,3:4]*alpha
      alpha *= (1-out[:,:,3:4])

  output += (1-mask)*bg
  return output, imgs, sublayers

def vq(img,table):
    e_img = tf.expand_dims(img,-2)
    _table = tf.reshape(table,[1,1,1,10,1])
    dist = tf.abs(e_img-table)
    dist = tf.reshape(dist,[num_mpi*reuse_fac*size_h*size_w,10])
    k = tf.argmin(dist, axis=-1)
    k = tf.reshape(k,[num_mpi*reuse_fac, size_h, size_w])
    z_q = tf.gather(table, k)
    return z_q

def train():
    lod_in = tf.placeholder(tf.float32, shape=[], name='lod_in')
    features = load_data()

    latent = np.random.uniform(-5,1,[laten_d, laten_h, laten_w, 1]).astype(np.float32)
    mpi = np.random.uniform(-3,1,[num_mpi*reuse_fac, size_h, size_w, 4]).astype(np.float32)
    mpi_c = np.random.uniform(-3,1,[num_mpi, size_h, size_w, 3]).astype(np.float32)
    mpi_a = np.random.uniform(-3,1,[num_mpi*reuse_fac, size_h, size_w, 1]).astype(np.float32)
    c_table = np.random.uniform(-3,1,[10,1]).astype(np.float32)


    bg = tf.get_variable("bg", initializer=np.array([1,1,1],dtype=np.float32), trainable=True)
    c_noise = tf.constant(1.1)
    #bg = bg + 0.5*c_noise*(2*tf.random_uniform(bg.shape)-1)* (1-lod_in/1200)
    bg = tf.sigmoid(bg)
    latent = tf.get_variable("depth", initializer=latent, trainable=True)
    latent = tf.sigmoid(latent)
    c_table = tf.get_variable("lookup_table", initializer=c_table, trainable=True)
    #latent = tf.concat([tf.sigmoid(latent[:,:,:,:3]), tf.sigmoid(latent[:,:,:,3:]*3)],-1)
    if reuse_fac == 1 :
        #mpi = tf.get_variable("mpi", initializer=mpi, trainable=True)
        mpi_c = tf.get_variable("mpic", initializer=mpi_c, trainable=True)
        mpi_aa = tf.get_variable("mpia", initializer=mpi_a, trainable=True)
        mpi_a = mpi_aa#vq(mpi_aa,c_table)
        #tf.add_to_collection("checkpoints", mpi_a)
        mpi = tf.concat([mpi_c,mpi_a],-1)
    else:
        mpi_c = tf.get_variable("mpic", initializer=mpi_c, trainable=True)
        mpi_c = tf.tile(mpi_c, [2, 1, 1, 1])
        mpi_a = tf.get_variable("mpia", initializer=mpi_a, trainable=True)
        mpi = tf.concat([mpi_c,mpi_a],-1)
    noise = c_noise*(2*tf.random_uniform(mpi.shape)-1) * (1-lod_in/1200)
    mpi = tf.sigmoid(mpi+noise)
    #mpia = tf.where(tf.random_uniform(mpi.shape) - mpip < 0, tf.ones_like(mpi), tf.zeros_like(mpi))
    #mpi = (mpip + mpia*3)/4


    img_out, shifts, sublayers = network(mpi, latent, bg, features, False)
    long = tf.concat(shifts, 1)
    tvc = tf.constant(0.0001) #0.0005
    tva = tf.constant(0.05) # 0.001

    with tf.compat.v1.variable_scope("loss"):
        mpiColor = mpi[:, :, :, :3]
        mpiAlpha = mpi[:, :, :, 3:4]
        mask = tf.reduce_mean(features["img"][0],-1,keepdims=True)
        mask = tf.cast(tf.greater(mask,0.015),tf.float32)
        loss =  100000 * tf.reduce_mean(mask*tf.square(img_out - features["img"][0]))
        loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
        loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
        #loss += 1 * tf.reduce_mean(tf.square(mpi_a*5 - tf.round(mpi_a*5) ))
        loss += 10 * tf.reduce_mean(tf.square(tf.sigmoid(mpi_a) - tf.round(tf.sigmoid(mpi_a)) ))
        #vq_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(mpi_aa), mpi_a) )
        #dq_loss = 0.25*tf.reduce_mean(tf.squared_difference(tf.stop_gradient(mpi_a), mpi_aa) )

    #long = tf.concat(shifts, 1)
    img_out = tf.clip_by_value(img_out,0.0,1.0)
    long = tf.clip_by_value(long,0.0,1.0)

    #lr = 0.1
    lr = tf.compat.v1.train.exponential_decay(0.1,lod_in,100,0.5)
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    t_vars = tf.trainable_variables()
    c_vars = [var for var in t_vars if 'mpic' in var.name]
    l_vars = [var for var in t_vars if 'lookup_table' in var.name]
    #print("l_var",l_vars)
    #l_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "mpia")
    #print("l_var",l_vars)
    nl_vars = [var for var in t_vars if 'lookup_table' not in var.name]
    #look_grad = tf.gradients(vq_loss,l_vars)
    #nlook_grad = tf.gradients(loss+dq_loss,nl_vars)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = slim.learning.create_train_op(loss,optimizer)
    train_op2 = slim.learning.create_train_op(loss,optimizer,variables_to_train=c_vars)

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
                    tf.compat.v1.summary.image("mpi_texture",mpi[2:3,:,:,:3]),
                    tf.compat.v1.summary.image("alpha",tf.expand_dims(long[:,:,3:4],0)),
                    tf.compat.v1.summary.image("color",tf.expand_dims(long[:,:,:3],0)),
                    tf.compat.v1.summary.image("depth",tf.expand_dims(long[:,:,4:5],0)),
                    ])

    saver = tf.train.Saver()
    if not os.path.exists('./model/' + FLAGS.dataset +'/'+ FLAGS.input):
        os.makedirs('./model/' + FLAGS.dataset +'/'+ FLAGS.input)
    for i in range(FLAGS.epoch+3):
        if i<200:
          _,los = sess.run([train_op,loss],feed_dict={lod_in:i})
        else:
          _,los = sess.run([train_op,loss],feed_dict={lod_in:i,tva:0.00001})
        if i%50==0:
            print(i, "loss = " ,los)
        if i%20 == 0:
           summ = sess.run(summary,feed_dict={lod_in:1200})
           writer.add_summary(summ,i)
        if i%200==0:
           saver.save(sess, './model/' + FLAGS.dataset +'/'+ FLAGS.input + '/' + str(i))
    """
    for i in range(203):
        _,los = sess.run([train_op2,loss],feed_dict={c_noise:0.1,lod_in:500,tvc:0.00001,tva:0.})
        if i%50==0:
            print(FLAGS.epoch+i, "Extra loss = " ,los)
        if i%20 == 0:
           summ = sess.run(summary,feed_dict={lod_in:1200})
           writer.add_summary(summ,FLAGS.epoch+i)
        if i%200==0:
           saver.save(sess, './model/' + FLAGS.dataset +'/'+ FLAGS.input + '/' + str(FLAGS.epoch+i))
    """

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
    #features = {}
    #features["r"] = tf.convert_to_tensor(np.array([ref_r]))
    #features["t"] = tf.convert_to_tensor(np.array([ref_t]))
    #features = tf.convert_to_tensor(features)


    latent = np.zeros([laten_d, laten_h, laten_w, 1],dtype=np.float32)

    latent[-2:laten_d,:,:,0] = 0.5
    for i in range(laten_d):
        ch, cw = int(laten_h/2-i/2), int(laten_w/2-i/4)
        latent[i,ch-10:ch+10,cw-5:cw+5,0] = 1.0

    mpi = np.zeros([num_mpi*reuse_fac,  size_h, size_w, 4],dtype=np.float32)
    mpi[0] = [1.,0.,0.,.95]
    mpi[1] = [1.,.5,0.,.95]
    mpi[2] = [1.,1.,0.,.95]
    mpi[3] = [.5,1.,0.,.95]

    mpi_c = np.random.uniform(-3,1,[num_mpi, size_h, size_w, 3]).astype(np.float32)
    mpi_a = np.random.uniform(-3,1,[num_mpi*reuse_fac, size_h, size_w, 1]).astype(np.float32)

    bg = tf.get_variable("bg", initializer=np.array([5,5,5],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)
    latent = tf.get_variable("depth", initializer=latent, trainable=False)
    latent = tf.sigmoid(latent)
    if (reuse_fac == 1):
        #mpi = tf.get_variable("mpi", initializer=mpi, trainable=True)
        mpi_c = tf.get_variable("mpic", initializer=mpi_c, trainable=True)
        mpi_a = tf.get_variable("mpia", initializer=mpi_a, trainable=True)
        #mpi_a = tf.round(mpi_a)
        #mpi_a = tf.where(tf.greater(mpi_a,0.5),5.*tf.ones_like(mpi_a),0.*tf.ones_like(mpi_a)) + tf.where(tf.greater(mpi_a,-1.5),0.*tf.ones_like(mpi_a),-5.*tf.ones_like(mpi_a))
        mpi = tf.concat([mpi_c,mpi_a],-1)
    else:
        mpi_c = tf.get_variable("mpic", initializer=mpi_c, trainable=True)
        mpi_c = tf.tile(mpi_c, [2, 1, 1, 1])
        mpi_a = tf.get_variable("mpia", initializer=mpi_a, trainable=True)
        mpi = tf.concat([mpi_c,mpi_a],-1)
    mpi = tf.sigmoid(mpi)




    img_out, shifts, sss = network(mpi, latent, bg, features, False)

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
        #plt.imshow(bug[:,:,:3])
        #tt = np.concatenate([bug[1,:,:,0],bug[20,:,:,0],bug[40,:,:,0],bug[59,:,:,0]],1)
        #plt.matshow(tt,cmap='gray')
        #plt.imshow(out)
        #plt.show()
        print("!!")
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

        cmd = 'ffmpeg -y -i ' + 'result/frame/\%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p result/' +FLAGS.input+FLAGS.version+ '.mp4'
        print(cmd)
        os.system(cmd)

    if True:
      webpath = "/var/www/html/orbiter/"
      if not os.path.exists(webpath + FLAGS.input+FLAGS.version):
          os.system("mkdir " + webpath + FLAGS.input+FLAGS.version)

      ret, sublay = sess.run([mpi,sss],feed_dict={features['r']:np.array([ref_r]),features['t']:np.array([ref_t])})
      sublayers = []
      mpis = []
      sublayers_combined = []
      print("sublay",sublay[0].shape,len(sublay))
      for i in range(num_mpi*reuse_fac):
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
      plt.imsave(webpath + FLAGS.input+FLAGS.version + "/mpi.png", mpis)
      plt.imsave(webpath + FLAGS.input+FLAGS.version + "/mpi_alpha.png", np.tile(mpis[:, :, 3:], (1, 1, 3)))
      sublayers = np.concatenate(sublayers, 1)
      sublayers = np.clip(sublayers, 0, 1)
      plt.imsave(webpath + FLAGS.input+FLAGS.version + "/sublayer.png", sublayers)
      sublayers_combined = np.concatenate(sublayers_combined, 1)
      plt.imsave(webpath + FLAGS.input+FLAGS.version + "/sublayers_combined.png", sublayers_combined)

      with open(webpath + FLAGS.input+FLAGS.version + "/extrinsics.txt", "w") as fo:
        for i in range(3):
          for j in range(3):
            fo.write(str(ref_r[i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t)]) + "\n")

      generateWebGL(webpath + FLAGS.input+FLAGS.version + "/index.html", w, h, getPlanes(),sub_sam, f, px, py)



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
