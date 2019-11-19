## 1 fixed voxel and multiple mpis model
## 2 times lookup (cubic depth and hi res mpi)
## for multiple mpis
## used feed_dict
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
tf.app.flags.DEFINE_boolean("predict", False, "making a video")
tf.app.flags.DEFINE_boolean("restart", False, "making a last video frame")
tf.app.flags.DEFINE_float("scale", 0.75, "scale input image by")
tf.app.flags.DEFINE_integer("subscale", 8, "downscale factor for the sub layer")

tf.app.flags.DEFINE_integer("layers", 25, "number of planes")
tf.app.flags.DEFINE_integer("sublayers", 2, "number of sub planes")
tf.app.flags.DEFINE_integer("epoch", 1000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_integer("index", 0, "index number")

tf.app.flags.DEFINE_string("dataset", "temple0", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "cen0", "input tfrecord")
tf.app.flags.DEFINE_string("ref_img", "cen0", "reference image")

#tf.app.flags.DEFINE_string("ref_img", "0051.png", "reference image such that MPI is perfectly parallel to")

sub_sam = max(FLAGS.sublayers,1)
num_mpi = FLAGS.layers
f = px = py = 0
ref_t = ref_c = ref_r = 0
offset = 32
dmin, dmax = -1, -1
num_ = 2
nh, nw = 0 , 0
img_ref = 0



def load_data(i,is_shuff=False):
  #filenames = ["sa0040","sa0045","sa0051","sa0057","sa0032","sa0033","sa0039"]
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

  # np.random.shuffle(filenames)
  #localpp = getLocalPath("/home2/suttisak","datasets/" + FLAGS.dataset + "/tem" + str(FLAGS.index%mpi_max) + ".train")
  localpp = "datasets/" + FLAGS.dataset + "/" + FLAGS.input + ".train"
  dataset = tf.data.TFRecordDataset([localpp])
  dataset = dataset.map(parser)
  if(is_shuff):  dataset = dataset.shuffle(5)
  dataset = dataset.repeat().batch(FLAGS.batch_size)

  return dataset.make_one_shot_iterator().get_next()

def setGlobalVariables():
  global f, px, py, ref_r, ref_t, ref_c, w, h, img_ref, nh, nw

  path = findCameraSfm(FLAGS.dataset)
  with open(path, "r") as f:
    js = json.load(f)


  f = float(js["intrinsics"][0]["pxFocalLength"]) * FLAGS.scale
  px = float(js["intrinsics"][0]["principalPoint"][0]) * FLAGS.scale
  py = float(js["intrinsics"][0]["principalPoint"][1]) * FLAGS.scale
  w = int(int(js["intrinsics"][0]["width"]) * FLAGS.scale)
  h = int(int(js["intrinsics"][0]["height"]) * FLAGS.scale)
  nh = h + offset * 2
  nw = w + offset * 2

  st = 0
  for view in js["views"]:
    if FLAGS.ref_img in view["path"]:
      for pose in js["poses"]:
        if pose["poseId"] == view["poseId"]:
          ref_r = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
          ref_t = -ref_r * np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
          ref_c = np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
          st += 1
          img_ref0 = plt.imread("datasets/" + FLAGS.dataset + "/undistorted/" + view["path"].split("/")[-1][:-3]+ "png")
          img_ref0 = cv2.resize(img_ref0, (w,h))
          break
      break
  img_ref = np.zeros((nh,nw,3),dtype=np.float32)
  img_ref[offset:h+offset,offset:w+offset,:] = img_ref0.copy()
  #cv2.imwrite("result/0.png",img_ref*255)
  #print(img_ref)
  #print("img_ref0",img_ref0.dtype)
  #print("img_ref",img_ref.dtype)
  #exit()

  global dmin, dmax
  if dmin <0 and dmax<0:
     with open("datasets/" + FLAGS.dataset + "/planes.txt", "r") as fi:
        dmin, dmax = [float(x) for x in fi.readline().split(" ")]

def computeHomography(r, t, d, ks=1,index=0):
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
  global nw, nh, offset
  #nh = h + offset * 2
  #nw = w + offset * 2

  H = computeHomography(r, t, d, ks)

  x, y = tf.meshgrid(list(range(w)), list(range(h)))
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)
  newCoords = tf.matmul(coords, tf.transpose(H))

  cx = tf.expand_dims(newCoords[:, :, 0] / newCoords[:, :, 2], 0)
  cy = tf.expand_dims(newCoords[:, :, 1] / newCoords[:, :, 2], 0)

  return bilinear_sampler(plane, cx + offset, cy + offset)

def getPlanes():
  if FLAGS.invz:
    return 1/np.linspace(1, 0.1, num_mpi) * dmin
  else:
    return np.linspace(dmin, dmax, num_mpi)

def network(mpi, rot,tra, is_training):
  alpha = 1
  output = 0
  mask = 0.0
  imgs = []
  sublayers = []
  planes = getPlanes()
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])
  
  mpi = tf.concat([mpi, tf.zeros_like(mpi[0:1])], 0)

  for i, v in enumerate(planes):
      aa = 1
      out = 0
      #if i>=num_mpi-1:
      #  break
      #depth = gen_depth(tf.concat([mpi[:,:,:,:,i],mpi[:,:,:,:,i+1]],-1),is_training)
      depth = gen_depth(mpi[i:i+2],is_training)
      depth = tf.image.resize(depth, [nh, nw], align_corners=True)
      for j in range(sub_sam):
          vv = j/sub_sam
          dep = rplanes[i]*(1-vv) + rplanes[i+1]*(vv)
          depth_map = depth[0,:,:,j:j+1]
          #img = samplePlane(tf.concat([mpi[0,:,:,:,i], depth_map], -1),rot,tra, dep, 1)
          img = samplePlane(tf.concat([mpi[i], depth_map], -1),rot,tra, dep, 1)
          #tf.compat.v1.add_to_collection("checkpoints", img)
          img = img[0]
          out += img[:,:,:4]*img[:,:,4:5]*aa
          aa  *= (1-img[:,:,4:5])
          depth_map = tf.image.resize(depth_map, [int(h/8), int(w/8)], tf.image.ResizeMethod.AREA)
          sublayers.append(depth_map)
          if j == 0:
              imgs.append(img)
      output += out[:,:,:3]*out[:,:,3:4]*alpha
      alpha *= (1-out[:,:,3:4])
      mask += out[:,:,3:4]*alpha

  #output += (1-mask)*bg
  return output, imgs, sublayers

def gen_mpi(input,is_train):
  with tf.variable_scope('gen',reuse=tf.AUTO_REUSE) :
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
        normalizer_params={"is_training": is_train}):
        layers = []
        size = []
        a, b = nh, nw
        next = input

        for chanel in [32,48,64,72]:
            layers.append(next)
            size.append((a,b))
            a, b = round((a+0.1)/2), round((b+0.1)/2)
            next = slim.conv2d(next,chanel,[3,3])
            next = slim.conv2d(next,chanel,[3,3],stride=2)  
        
        #next = tf.tile(next,[num_mpi,1,1,1])
        #nois = tf.compat.v1.get_variable("source",shape=[num_mpi,a,b,3], trainable=True)
        #next = tf.concat([next,nois],-1)      

        for i in range(3):
          next = next + slim.conv2d(next,72,[3,3])

        for i, lay in reversed(list(enumerate(layers))):
            next = tf.image.resize(next,size[i])
            #next = tf.concat([next,tf.tile(lay,[num_mpi,1,1,1])],-1)
            next = tf.concat([next,lay],-1)
            next = slim.conv2d(next,72,[3,3])
            next = next + slim.conv2d(next,72,[3,3])
        
        next = slim.conv2d(next,3*num_mpi,[3,3])
        next = tf.reshape(next,(1,nh,nw,3,num_mpi))
        nois = tf.compat.v1.get_variable("alpha0",shape=[1,nh,nw,1,num_mpi], trainable=True)
        next = tf.concat([next,nois],3)
        #next = next + slim.conv2d(next,4,[3,3],activation_fn=tf.sigmoid)
    return next

def gen_mpi2(is_train):
  with tf.variable_scope('gen',reuse=tf.AUTO_REUSE) :
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
        normalizer_params={"is_training": is_train}):
        layers = []
        size = []
        a, b = nh, nw
        for i in range(2):
            size.append((a,b))
            a, b = round((a+0.1)/2), round((b+0.1)/2) 

        rgb = tf.compat.v1.get_variable("rgb",shape=[num_mpi,a,b,3], trainable=True)    
        alpha = tf.compat.v1.get_variable("alpha",shape=[num_mpi,a,b,1], trainable=True)   

        #rgb = slim.conv2d(rgb,48,[3,3])    
        #alpha = slim.conv2d(alpha,16,[3,3])    

        for i, ss in reversed(list(enumerate(size))):
            rgb = tf.image.resize(rgb,ss)
            alpha = tf.image.resize(alpha,ss)
            rgb0 = rgb
            alpha0 = alpha
            for chanel in [4, 8, 12, 16]:
              rgb = slim.conv2d(rgb,3*chanel,[3,3])
              alpha = slim.conv2d(alpha,chanel,[3,3])
            rgb = slim.conv2d(rgb,3,[3,3],activation_fn=None)
            alpha = slim.conv2d(alpha,1,[3,3],activation_fn=None)
            rgb = rgb0 + rgb
            alpha = alpha0 + alpha

        #rgb = slim.conv2d(rgb,3,[3,3])
        #alpha = slim.conv2d(alpha,1,[3,3])
        mpi = tf.concat([rgb,alpha],3)
        #mpi = mpi + slim.conv2d(mpi,4,[3,3],activation_fn=None)
    return tf.sigmoid(mpi)

def gen_depth(input,is_train,reuse=False):
    with tf.variable_scope('depth',reuse=tf.AUTO_REUSE) :
    #with tf.compat.v1.variable_scope('depth') as scope:
    #    if reuse: scope.reuse_variables()

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

def discriminator(input,is_train):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
        normalizer_params={"is_training": is_train}):

        next = slim.conv2d(input,16,[3,3],stride=2)
        for chanel in [32,48,64]:
            next = slim.conv2d(next,chanel,[3,3])
            next = slim.conv2d(next,chanel,[3,3])

        next = slim.conv2d(next,1,[3,3],activation_fn=tf.sigmoid)
    return next

def Multi_discriminator(input,is_train,reuse=False):
  input = tf.expand_dims( input,0)
  with tf.variable_scope('dis') as scope:
      if reuse: scope.reuse_variables()

      out1 = discriminator(input,is_train)
      out1 = tf.image.resize(out1,[int(h/2),int(w/2)])
      out2 = discriminator(tf.image.resize(input,[int(h/2),int(w/2)]),is_train)
      out2 = tf.image.resize(out2,[int(h/2),int(w/2)])
      out3 = discriminator(tf.image.resize(input,[int(h/4),int(w/4)]),is_train)
      out3 = tf.image.resize(out3,[int(h/2),int(w/2)])

  return out1*0.7+ out2*0.2 + out3*0.1

def train():
    global nh, nw, img_ref
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    rot = tf.compat.v1.placeholder(tf.float32, shape=[3,3], name='rotation')
    tra = tf.compat.v1.placeholder(tf.float32, shape=[3,1], name='translation')
    real_img = tf.compat.v1.placeholder(tf.float32, shape=[h,w,3], name='ex_img')

    #img_reff = tf.convert_to_tensor(img_ref)
    #img_reff = tf.expand_dims(img_reff,0)

    #lr = 0.1
    lr = tf.compat.v1.train.exponential_decay(0.1,lod_in,1000,0.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    #optimizer = tf.train.RMSPropOptimizer(lr)
    with tf.compat.v1.variable_scope("Net"):
        #mpi = gen_mpi(img_reff,True)
        mpi = gen_mpi2(True)
        img_out, shifts, sss = network(mpi, rot,tra, True)
        long = tf.concat(shifts, 1)

        real_result = Multi_discriminator(real_img,True)
        fake_result = Multi_discriminator(img_out,True, reuse=True)


    with tf.compat.v1.variable_scope("loss"):
        tva = tf.constant(0.1) *0.05  #*0.05
        tvc = tf.constant(0.005) *0.05#*0.15
        avl = tf.constant(0.01) 
        mpiColor = mpi[:, :, :, :3]
        mpiAlpha = mpi[:, :, :, 3:4]
        loss =  100000 * tf.reduce_mean(tf.square(img_out - real_img))
        #for i in range(num_mpi):
        #  loss += tva/num_mpi * tf.reduce_mean(tf.image.total_variation(mpi[:, :, :, 3:4,i]))
        loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
        #loss += tva * 0.25 * tf.reduce_mean(mpiAlpha*2 - tf.round(mpiAlpha*2))
        #loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
        #loss += avl * tf.reduce_mean(tf.image.total_variation(depth0))
        #loss += avl * tf.reduce_mean(abs(depth0[:,:,:,:-1] - depth0[:,:,:,1:]) )
        g_loss = tf.reduce_mean(tf.square(fake_result))
        #loss += avl * g_loss

        d_loss = tf.reduce_mean(tf.square(fake_result-1)) + tf.reduce_mean(tf.square(real_result))

    t_vars = tf.compat.v1.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'dis' not in var.name]

    with tf.compat.v1.variable_scope("post"):
        image_out = tf.clip_by_value(img_out,0.0,1.0)
        train_op = slim.learning.create_train_op(loss,optimizer,variables_to_train=g_vars)
        train_dis = slim.learning.create_train_op(d_loss,optimizer,variables_to_train=d_vars)
        train_gen = slim.learning.create_train_op(g_loss,optimizer,variables_to_train=g_vars)
        
        long = tf.clip_by_value(long,0.0,1.0)

        summary = tf.compat.v1.summary.merge([
                        tf.compat.v1.summary.scalar("all_loss", loss),
                        #tf.compat.v1.summary.scalar("dis_loss", d_loss),
                        #tf.compat.v1.summary.scalar("gen_loss", g_loss),

                        tf.compat.v1.summary.image("out",tf.expand_dims(image_out,0)),
                        tf.compat.v1.summary.image("origi",tf.expand_dims(real_img,0)),
                        tf.compat.v1.summary.image("alpha",tf.expand_dims(long[:,:,3:4],0)),
                        tf.compat.v1.summary.image("color",tf.expand_dims(long[:,:,:3]*long[:,:,3:4],0)),
                        #tf.compat.v1.summary.image("map_areal",real_result),
                        #tf.compat.v1.summary.image("map_fake",tf.concat([real_result,fake_result],2)),
                        #tf.compat.v1.summary.image("map_fake",fake_result),
                        #tf.compat.v1.summary.image("map_fake",tf.concat([real_result,fake_result],2)),
                        ])

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
    localpp = './model/' + FLAGS.dataset +'/tem'+str(FLAGS.index)
    if not os.path.exists(localpp):
        os.makedirs(localpp)

    features = load_data(0,is_shuff = True)
    sess = tf.compat.v1.Session(config=config)
    if 0:
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        ckpt = tf.train.latest_checkpoint(localpp )
        saver.restore(sess, ckpt)
    else:
        if not FLAGS.restart:
          variables_to_restore = slim.get_variables_to_restore()
          saver = tf.train.Saver(variables_to_restore)
          ckpt = tf.train.latest_checkpoint(localpp )
          saver.restore(sess, ckpt)
        else:
          sess.run(tf.compat.v1.global_variables_initializer())

    rot1 = ref_r
    tra1 = ref_t
    for i in range(FLAGS.epoch + 3):
        #feed = sess.run(features)
        for j in range(1):
          feed = sess.run(features)
          feedlis = {lod_in:i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]}
          #_,dlos = sess.run([train_dis,d_loss],feed_dict=feedlis)
          #rand = np.random.rand()*1.2 - 0.1
          #rot1 = interpolate_rotation(rot1, feed["r"][0], rand)
          #tra1 = tra1 * (1-rand) + feed["t"][0] * rand
          #if j%2==1:
          #  feedlis1 = {lod_in:i,rot:rot1,tra:tra1,real_img:feed["img"][0]}
          #  _,glos = sess.run([train_gen,g_loss],feed_dict=feedlis1)

        if i < 6000:
            _,los = sess.run([train_op,loss],feed_dict=feedlis)
        else:
            _,los = sess.run([train_op,loss],feed_dict={lod_in:600,avl:1000,tvc:0.0,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
        
        #for j in range(5):
        #    _,dlos = sess.run([train_dis,d_loss],feed_dict={lod_in:600,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
    
        if i%100==0:
            print(FLAGS.index,i, "loss = " ,los)
            plt.imsave('result/0.png',sess.run(image_out,feed_dict={rot:feed["r"][0],tra:feed["t"][0]}))
        if i%20 == 0:
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


    nh = h + offset * 2
    nw = w + offset * 2

    depth_init = np.random.uniform(-5,0,[num_mpi, int(h/ FLAGS.subscale), int(w/ FLAGS.subscale), sub_sam]).astype(np.float32)

    #mpi = np.zeros([num_mpi, nh, nw, 4],dtype=np.float32)
    #mpi[0] = [1.,0.,0.,.95]
    #mpi[1] = [1.,.5,0.,.95]
    #mpi[2] = [1.,1.,0.,.95]
    #mpi[3] = [.5,1.,0.,.95]

    #depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
    #depth = tf.sigmoid(depth)
    #depth = tf.image.resize(depth, [h, w], align_corners=True)
    #img_reff = tf.convert_to_tensor(img_ref)
    #img_reff = tf.expand_dims(img_reff,0)


    with tf.compat.v1.variable_scope("Net"):
        #mpi = gen_mpi(img_reff,False)
        mpi = gen_mpi2(False)
        img_out, shifts, sss = network(mpi,rot,tra, False)
        long = tf.concat(shifts, 1)

    with tf.compat.v1.variable_scope("post"):
        image_out= tf.clip_by_value(img_out,0.0,1.0)
        long = tf.clip_by_value(long,0.0,1.0) 

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    #localpp = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index)%mpi_max))
    localpp = './model/' + FLAGS.dataset +'/tem'+str((FLAGS.index))
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)


    if True:
        for i in range(0,300,1):
          feed = sess.run(features)
          if(i%50==0): print(i)
          out = sess.run(image_out,feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
          plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%( i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p result/'+FLAGS.dataset+'_moving.mp4'
        print(cmd)
        os.system(cmd)

    #feed = sess.run(features)
    #plt.imsave('result/0.png',sess.run(long[:,:,:3]*long[:,:,3:4],feed_dict={rot:feed["r"][0],tra:feed["t"][0]}))

    if False:
      webpath = "webpath/"  #"/var/www/html/orbiter/"
      if not os.path.exists(webpath + FLAGS.dataset):
          os.system("mkdir " + webpath + FLAGS.dataset)


      for ii in range(1):
        ref_rt = np.array(ref_r)
        ref_tt = np.array(ref_t)
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
        plt.imsave(webpath + FLAGS.dataset+ "/mpi%02d.png"%(FLAGS.index+ii), mpis)
        #plt.imsave(webpath + FLAGS.dataset+ "/mpi_alpha"+str(ii)+".png", np.tile(mpis[:, :, 3:], (1, 1, 3)))
        sublayers = np.concatenate(sublayers, 1)
        sublayers = np.clip(sublayers, 0, 1)
        plt.imsave(webpath + FLAGS.dataset+ "/sublayer%02d.png"%(FLAGS.index+ii), sublayers)
        #sublayers_combined = np.concatenate(sublayers_combined, 1)
        #plt.imsave(webpath + FLAGS.dataset+ "/sublayers_combined.png", sublayers_combined)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi.png", mpis)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi_alpha.png", np.tile(mpis[:, :, 3:], (1, 1, 3)))

      namelist = "["
      for ii in range(FLAGS.index+1):
        namelist += "\"%02d\","%(ii)
      namelist += "]"
      print(namelist)

      with open(webpath + FLAGS.dataset+ "/extrinsics%02d.txt"%(FLAGS.index), "w") as fo:
        for i in range(3):
          for j in range(3):
            fo.write(str(ref_r[ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t)]) + "\n")

      #generateWebGL(webpath + FLAGS.dataset+ "/index.html", w, h, getPlanes(),namelist,sub_sam, f, px, py)

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
