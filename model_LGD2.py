## faster render equation
## for multiple resolution
## inverse transformation
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.compat.v1 import ConfigProto
from mpl_toolkits.mplot3d import Axes3D
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

from view_gen import generateWebGL, generateConfigGL
from utils import *
from localpath import getLocalPath

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("invz", False, "using inverse depth, ignore dmax in this case")
tf.app.flags.DEFINE_boolean("predict", False, "making a video")
tf.app.flags.DEFINE_boolean("restart", False, "making a last video frame")
tf.app.flags.DEFINE_float("scale", 0.75, "scale input image by")
tf.app.flags.DEFINE_integer("offset", 64, "offset size to mpi")
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
offset = FLAGS.offset
dmin, dmax = -1, -1
num_ = 1
nh, nw = 0 , 0
img_ref = 0
is_gen_mpi = False
is_gen_depth = False

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

def getPlanes(mode=0):
    if False:
      return (dmax-dmin)*np.linspace(0, 1, num_mpi)**4+dmin
    elif FLAGS.invz:
      return 1/np.linspace(1, dmin/dmax, num_mpi) * dmin
    else:
      return np.linspace(dmin, dmax, num_mpi)

def network(mpi,depth, rot,tra, is_training):
  alpha, output = 1, 0
  imgs = []
  sublayers = []
  planes = getPlanes()
  print(planes)
  mask = 1
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])
  mpic, mpia = mpi[:,:,:,:3], mpi[:,:,:,3:4]
  x, y = tf.meshgrid(list(range(w)), list(range(h)))
  x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)

  for i, v in enumerate(planes):
      for j in range(sub_sam):
        vv = j/sub_sam
        if FLAGS.invz:
          dep = 1/(1/rplanes[i]*(1-vv) + 1/rplanes[i+1]*(vv))
        else:
          dep = rplanes[i]*(1-vv) + rplanes[i+1]*(vv)

        H = computeHomography(rot, tra, dep)
        newCoords = tf.matmul(coords, tf.transpose(H))
        cxy = tf.expand_dims(newCoords[:, :, :2] / newCoords[:, :, 2:3], 0) + offset
        warp = cxy if j==0 else tf.concat([warp,cxy],0)
        if i*j==0 : mask *= tf.contrib.resampler.resampler(tf.ones_like(mpia[i:i+1]),cxy)
        if i+1==num_mpi and j+1==sub_sam : mask *= tf.contrib.resampler.resampler(tf.ones_like(mpia[i:i+1]),cxy)

      a1 =  tf.transpose(depth[i:i+1], perm=[3,1,2,0])
      a2 =  mpia[i:i+1] * a1
      a3 =  mpic[i:i+1] * a2
      img1 = tf.contrib.resampler.resampler(a1,warp)
      img2 = tf.contrib.resampler.resampler(a2,warp)
      img3 = tf.contrib.resampler.resampler(a3,warp)
      weight = tf.cumprod(1 - img1,0,exclusive=True)
      output += tf.reduce_sum(weight*img3,0,keepdims=True)*alpha
      alpha_mul = 1 - tf.reduce_sum(weight*img2,0,keepdims=True)
      alpha *= alpha_mul
      if i==1: com_alpha = calpha*alpha_mul
      if i>1: com_alpha = tf.concat([com_alpha,calpha*alpha_mul],0)
      calpha = alpha_mul
  if is_training:    
    output += (tf.random.uniform([1,1,1,3],maxval=0.1))*alpha

  return output, com_alpha, mask

def samplePlane(img, rot,tra):
  alpha, output = 1, 0
  imgs = []
  sublayers = []
  planes = getPlanes()
  mask = 1
  rplanes = np.concatenate([planes, 2*planes[-1:]-planes[-2:-1]])
  
  x, y = tf.meshgrid(list(range(nw)), list(range(nh)))
  x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)
  img_tile = tf.tile(tf.expand_dims(img,0),[num_mpi,1,1,1])

  for i, dep in enumerate(planes):

      H = computeHomography(rot, tra, dep)
      newCoords = tf.matmul(coords, tf.transpose(tf.linalg.inv(H)))
      cxy = tf.expand_dims(newCoords[:, :, :2] / newCoords[:, :, 2:3], 0) - offset
      warp = cxy if i==0 else tf.concat([warp,cxy],0)

  inv_img = tf.contrib.resampler.resampler(img_tile,warp)

  return inv_img 

def conv_grad(in_grad,mpi0,is_train = True):
    #Try to clone DeepView
    with tf.compat.v1.variable_scope("block1", reuse=tf.compat.v1.AUTO_REUSE):
        #o1 = samplePlane(img_ref,ref_r,ref_t)
        o1 = in_grad[0]
        #o1 = lrelu(conv2d(o1,16,stride=[1,2,2,1],name='c0'))
        o1 = lrelu(conv2d(o1,8,name='c0'))
        o1 = tf.nn.space_to_depth(o1,2)
        o1 = lrelu(conv2d(o1,32,name='c1'))
        o1 = lrelu(conv2d(o1,64,name='c2'))
        o1 = lrelu(conv2d(o1,24,name='c3'))
        o1 = lrelu(conv2d(o1,24,name='c4'))
        o1 = lrelu(conv2d(o1,32,filter=1,name='c5'))
        k = tf.get_variable("mem_max"+str(0), [num_mpi,int(nh/2),int(nw/2),32],trainable=False)
        tf.compat.v1.add_to_collection("mem_max",k)
        m = tf.maximum(o1,k)
        tf.compat.v1.add_to_collection("maxi",m)

    with tf.compat.v1.variable_scope("block2", reuse=tf.compat.v1.AUTO_REUSE):
        o1 = tf.concat([o1,m],-1)
        o1 = lrelu(conv2d(o1,64,filter=1,name='c1'))
        o1 = lrelu(conv2d(o1,32,filter=1,name='c2'))
        k = tf.get_variable("mem_max"+str(1), [num_mpi,int(nh/2),int(nw/2),32],trainable=False)
        tf.compat.v1.add_to_collection("mem_max",k)
        m = tf.maximum(o1,k)
        tf.compat.v1.add_to_collection("maxi",m)

    with tf.compat.v1.variable_scope("block3", reuse=tf.compat.v1.AUTO_REUSE):
        o1 = tf.concat([o1,m],-1)
        #o1 = lrelu(conv2d(o1,16,stride=[1,2,2,1],filter=1,name='c1'))
        o1 = lrelu(conv2d(o1,64,filter=1,name='c1'))
        o1 = lrelu(conv2d(o1,32,filter=1,name='c2'))
        k = tf.get_variable("mem_max"+str(2), [num_mpi,int(nh/2),int(nw/2),32],trainable=False)
        tf.compat.v1.add_to_collection("mem_max",k)
        m = tf.maximum(o1,k)
        tf.compat.v1.add_to_collection("maxi",m)
        #m = upscale2d(m)

    with tf.compat.v1.variable_scope("block4", reuse=tf.compat.v1.AUTO_REUSE):
        mpi = lrelu(conv2d(mpi0,8,name='c0'))
        mpi = tf.nn.space_to_depth(mpi,2)
        mpi = tf.concat([mpi,m],-1)
        mpi = lrelu(conv2d(mpi,16,name='c1'))
        mpi = lrelu(conv2d(mpi,16,name='c2'))
        mpi = tf.nn.depth_to_space(mpi,2)
        mpi = conv2d(mpi,4,name='c3')

    return mpi

def vgg_layers(layer_names):
    global h,w, offset
    nh = h + offset * 2
    nw = w + offset * 2
    with tf.variable_scope('vgg', reuse=tf.compat.v1.AUTO_REUSE):
        vgg = tf.keras.applications.VGG16(include_top=False, weights  = 'imagenet', input_shape = (nh, nw, 3))
        vgg.trainable = False
        #vgg.summary()
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
    return model

def vgg_loss(gen_pic, ground_truth):
    global h, w
    nh = h + offset * 2
    nw = w + offset * 2
    layer_name = ['block2_conv2','block3_conv3']#['input_1']
    #with tf.compat.v1.variable_scope("loss", reuse=tf.compat.v1.AUTO_REUSE):
    vgg_intermidiate = vgg_layers(layer_name)
    #gen_pic = tf.image.resize(gen_pic, (224, 224))
    #ground_truth  = tf.image.resize(ground_truth, (224, 224))
    gen_pic = tf.keras.applications.vgg16.preprocess_input(gen_pic*255)
    ground_truth = tf.keras.applications.vgg16.preprocess_input(ground_truth*255)
    intermidiate_layer_gen = vgg_intermidiate(gen_pic)
    intermidiate_layer_gt = vgg_intermidiate(ground_truth)
    loss = 0
    for i in range(len(layer_name)):
        loss += tf.reduce_mean(tf.square(intermidiate_layer_gen[i] - intermidiate_layer_gt[i] ))#- intermidiate_layer_gt[i])) #intermidiate_layer_img2[i]))
    print(tf.compat.v1.trainable_variables())

    return loss

def train(logres=4):
    global nh, nw, img_ref
    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    
    features = load_data(FLAGS.dataset,FLAGS.input,[h,w],2,is_shuff = True)

    rot = features['r']
    tra = features['t']
    real_img = features['img']

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, nh, nw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi, nh, nw, 1]).astype(np.float32)
    depth_init = np.random.uniform(-5,0,[num_mpi, int(nh/ FLAGS.subscale), int(nw/ FLAGS.subscale), sub_sam]).astype(np.float32)

    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
        mpic = tf.random.uniform([num_mpi, nh, nw, 3], -1, 1, name = "mpi_c")
        #mpic = -tf.log(1/(samplePlane(img_ref,ref_r,ref_t)+0.001)-1)
        mpia = tf.random.uniform([num_mpi, nh, nw, 1], -5,-3, name = "mpi_a")
        mpi0 = tf.concat([mpic,mpia],-1)

        mpi_sig = tf.sigmoid(mpi0)

        if is_gen_depth:
            if FLAGS.sublayers<1: depth = tf.get_variable("Net_depth", initializer=np.ones((num_mpi,3,3,1)), trainable=False)
            else: depth = gen_depth(mpi,True)
        else:
            depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
            depth = tf.sigmoid(depth)
            depth = tf.image.resize(depth, [nh, nw], align_corners=True)

    layer_name = ['block2_conv2','block3_conv3']
    vgg_intermidiate = vgg_layers(layer_name)


    loss = 0
    #with tf.compat.v1.variable_scope("loss%d"%(FLAGS.index)):
    img_out, allalpha, mask = network( mpi_sig, depth, rot[0],tra[0], False)
    loss +=  100000 * tf.reduce_mean(tf.square(img_out - real_img[0])*mask)
    #loss += 100000 * vgg_loss(tf.expand_dims(img_out,0), real_img[0:1]) 
    #og_pic0 = tf.keras.applications.vgg16.preprocess_input(real_img[0:1]*255)
    #gen_pic0 = tf.keras.applications.vgg16.preprocess_input(img_out*255)
    #im_layer_gen_pic0 = vgg_intermidiate(gen_pic0)
    #im_layer_og_pic0 = vgg_intermidiate(og_pic0)
    #for i in range(len(layer_name)):
    #    loss += tf.reduce_mean(tf.square(im_layer_gen_pic0[i] - im_layer_og_pic0[i] ))
    fac = 1.0 #(1 - iter/(1500*2)) *0.01
    tva = tf.constant(0.1) * fac #*0.01
    tvc = tf.constant(0.005)  * fac *0.0
    mpiColor = mpi_sig[:, :, :, :3]
    mpiAlpha = mpi_sig[:, :, :, 3:4]
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
    grad_mpi = tf.gradients(loss,mpi0)

    new_grad = conv_grad(grad_mpi,mpi0,True)
    new_mpi = tf.sigmoid(mpi0 + new_grad)
    new_img, allalpha, mask = network( new_mpi, depth, rot[-1],tra[-1], False)
    new_loss = 0
    new_loss +=  100000 * tf.reduce_mean(tf.square(new_img - real_img[-1])*mask)
    #new_loss += 100000 * vgg_loss(tf.expand_dims(new_img,0), real_img[1:2]) 
    new_loss += tvc * tf.reduce_mean(tf.image.total_variation (new_mpi[:, :, :, :3]))
    new_loss += tva * tf.reduce_mean(tf.image.total_variation (new_mpi[:, :, :, 3:4]))

    gra_mpi = tf.sigmoid(mpi0 - 0.1*grad_mpi[0])
    gra_img, allalpha, mask = network( gra_mpi, depth, rot[-1],tra[-1], False)
    gra_loss = 0
    gra_loss +=  100000 * tf.reduce_mean(tf.square(gra_img - real_img[-1])*mask)
    #gra_loss += 100000 * vgg_loss(tf.expand_dims(gra_img,0), real_img[1:2]) 
    gra_loss += tvc * tf.reduce_mean(tf.image.total_variation (gra_mpi[:, :, :, :3]))
    gra_loss += tva * tf.reduce_mean(tf.image.total_variation (gra_mpi[:, :, :, 3:4]))


    image_out = tf.clip_by_value(img_out,0.0,1.0)
    long = tf.reshape(mpi_sig,(1,num_mpi*nh,nw,4))
    image2_out = tf.clip_by_value(new_img,0.0,1.0)
    long2 = tf.reshape(new_mpi,(1,num_mpi*nh,nw,4))
    image3_out = tf.clip_by_value(gra_img,0.0,1.0)
    long3 = tf.reshape(gra_mpi,(1,num_mpi*nh,nw,4))
    #lr = 0.1
    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.2)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    t_vars = tf.compat.v1.trainable_variables()
    l_vars = [var for var in t_vars if 'vgg' not in var.name]
    #print(t_vars)
    #print([var for var in t_vars if 'block' not in var.name])
    #exit()

    train_op = slim.learning.create_train_op(new_loss,optimizer,variables_to_train=l_vars)


    name1 = tf.compat.v1.get_collection('mem_max')
    name2 = tf.compat.v1.get_collection('maxi')
    memmax = [v.assign(name2[i]) for i, v in enumerate(name1)]
    clmem = [v.assign(v*0.0) for i, v in enumerate(name1)]

    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/old_loss", gra_loss),
                tf.compat.v1.summary.scalar("post0/new_loss", new_loss),
                #tf.compat.v1.summary.image("post0/out",tf.expand_dims(tf.concat([real_img[-1],image_out],0),0)),
                tf.compat.v1.summary.image("post0/out2",tf.concat([real_img[1:],image2_out],1)),
                tf.compat.v1.summary.image("post0/out3",tf.concat([real_img[1:],image3_out],1)),
                #tf.compat.v1.summary.image("post1/o_alpha",long[:,:,:,3:4]),
                #tf.compat.v1.summary.image("post1/o_color",long[:,:,:,:3]),
                tf.compat.v1.summary.image("post1/n_alpha",long2[:,:,:,3:4]),
                tf.compat.v1.summary.image("post1/n_color",long2[:,:,:,:3]),
                tf.compat.v1.summary.image("post1/g_alpha",long3[:,:,:,3:4]),
                tf.compat.v1.summary.image("post1/g_color",long3[:,:,:,:3]),
                ])

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    localpp = "TensorB/"+FLAGS.dataset+"_s%02d"%FLAGS.subscale
    if FLAGS.index==0:
      if os.path.exists(localpp):
        os.system("rm -rf " +localpp )
    if not os.path.exists(localpp):
      os.makedirs(localpp)
    writer = tf.compat.v1.summary.FileWriter(localpp)
    #writer.add_graph(sess.graph)

    saver = tf.train.Saver()
    localpp = './model/' + FLAGS.dataset +"/s%02d"%FLAGS.subscale
    if not os.path.exists(localpp):
        os.makedirs(localpp)

    
    sess = tf.compat.v1.Session(config=config)
    #sess.run(tf.compat.v1.global_variables_initializer())
    if not FLAGS.restart:
      variables_to_restore = slim.get_variables_to_restore()
      t_vars = slim.get_variables_to_restore()
      var2restore = [var for var in t_vars if 'mpi_' in var.name ]
      print(var2restore)
      saver = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.latest_checkpoint(localpp )
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.compat.v1.global_variables_initializer())
      #sess.run(tf.variables_initializer(l_vars))

    for i in range(FLAGS.epoch + 3):
        if i<100+50:  nn = 3 - max((i-100)/50,0)
        elif i<200+50:  nn = 2 - max((i-200)/50,0)
        elif i<600+50:  nn = 1 - max((i-600)/50,0)
        else: nn = 0 
        nn += (np.random.rand()*3 - 1.5)
        feedlis = {iter:i,lod_in:nn}

        _,_,los,los2 = sess.run([train_op,memmax,new_loss,gra_loss],feed_dict=feedlis)

        if i%100==0:
            print(FLAGS.index,i, "loss = " ,los, los2)
        if i%20 == 0:
            summ = sess.run(summary,feed_dict=feedlis)
            writer.add_summary(summ,i)
        if i%4 == 0:
            sess.run(clmem)
            sess.run(memmax,feed_dict=feedlis)
        #if i%50 == 49:
        #    sess.run(upres,feed_dict=feedlis)
        if i%200==1:
            saver.save(sess, localpp + '/' + str(000))


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

    depth_init = np.random.uniform(-5,0,[num_mpi, int(nh/ FLAGS.subscale), int(nw/ FLAGS.subscale), sub_sam]).astype(np.float32)

    int_noise = np.random.uniform(-1,-1,[num_mpi, int(nh/ 2), int(nw/ 2), 4]).astype(np.float32)

    mpi = np.zeros([num_mpi, nh, nw, 4],dtype=np.float32)
    mpi[0] = [1.,0.,0.,.95]
    mpi[1] = [1.,.5,0.,.95]
    mpi[2] = [1.,1.,0.,.95]
    mpi[3] = [.5,1.,0.,.95]

    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
        if is_gen_mpi:
          noise = tf.compat.v1.get_variable("noise", initializer=int_noise, trainable=False)
          mpic = gen_mpi(noise,lod_in,is_train=True)
        if False:
          mpic = tf.compat.v1.get_variable("mpi_c", initializer=mpi[:,:,:,:3], trainable=False)   
          mpia1 = tf.compat.v1.get_variable("mpi_a1", initializer=mpi[:num_mpi-1,:,:,3:4], trainable=False)
          mpia2 = tf.compat.v1.get_variable("mpi_la", initializer=mpi[:1,:,:,3:4]*0+5, trainable=False)
        
        #mpic = tf.compat.v1.get_variable("mpi_c", initializer=mpi[:,:,:,:3], trainable=False)   
        mpi = gen_mpi(features,lod_in,False)
        #mpia = tf.compat.v1.get_variable("mpi_a", initializer=mpi[:num_mpi,:,:,3:4], trainable=False)
        #mpi = tf.concat([mpic,mpia],-1)

        mpi = tf.sigmoid(mpi)
        if is_gen_depth:
          if FLAGS.sublayers<1: depth = tf.get_variable("Net_depth", initializer=np.ones((num_mpi,3,3,1)), trainable=False)
          else: depth0 = gen_depth(mpi,True)
        else:
          depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
          #depth2 = tf.get_variable("depth2", initializer=depth_init[num_mpi-1:,:,:,:sub_sam-1], trainable=True)
          #depth3 = tf.get_variable("depth3", initializer=depth_init[num_mpi-1:,:,:,:1]*0+5, trainable=False)
          #depth = tf.concat([depth2,depth3],-1)
          #depth = tf.concat([depth1,depth],0)
          #depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
          depth0 = tf.sigmoid(depth)
        depth = tf.image.resize(depth0, [nh, nw], align_corners=True)
        img_out,  allalpha, mask = network(mpi, depth, rot,tra, False)

    with tf.compat.v1.variable_scope("post%d"%(FLAGS.index)):
        image_out= tf.clip_by_value(img_out,0.0,1.0)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore()
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    localpp = './model/' + FLAGS.dataset +"/s%02d"%FLAGS.subscale
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)


    if True:  # make sample picture and video
        webpath = "webpath/"  #"/var/www/html/orbiter/"
        if not os.path.exists(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale):
            os.system("mkdir " + webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale)

        for i in range(0,300,1):
          feed = sess.run(features)
          out = sess.run(image_out,feed_dict={lod_in:0,rot:feed["r"][0],tra:feed["t"][0]})
          if(i%50==0): 
            print(i)
            plt.imsave("webpath/"+FLAGS.dataset+"_s%02d"%FLAGS.subscale+"/%04d.png"%( i),out)
          plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%( i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p webpath/'+FLAGS.dataset+"_s%02d"%FLAGS.subscale+'/moving.mp4'
        print(cmd)
        os.system(cmd)

    if True:  # make web viewer
      webpath = "webpath/"  #"/var/www/html/orbiter/"
      if not os.path.exists(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale):
          os.system("mkdir " + webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale)

      ret, sublay = sess.run([mpi,depth0],feed_dict={lod_in:0})
      sublayers = []
      mpis = []
      sublayers_combined = []
      for i in range(num_mpi):
          mpis.append(ret[i,:,:,:4])
          ls = []
          for j in range(sub_sam):
              ls.append(sublay[i,:,:,j:j+1])
          ls = np.concatenate(ls,0)
          ls = np.tile(ls,(1,1,3))
          sublayers.append(ls)
          ls = np.reshape(ls, (sub_sam,sublay[0].shape[0],sublay[0].shape[1],3))
          ls = np.clip(np.sum(ls, 0), 0.0, 1.0)
          sublayers_combined.append(ls)

      mpis = np.concatenate(mpis, 1)
      plt.imsave(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/mpi%02d.png"%(FLAGS.index), mpis)
      sublayers = np.concatenate(sublayers, 1)
      sublayers = np.clip(sublayers, 0, 1)
      plt.imsave(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/sublayer%02d.png"%(FLAGS.index), sublayers)

      plt.imsave(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/mpi.png", mpis)
      plt.imsave(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/mpi_alpha.png", np.tile(mpis[:, :, 3:], (1, 1, 3)))

      namelist = "["
      for ii in range(FLAGS.index+1):
        namelist += "\"%02d\","%(ii)
      namelist += "]"
      print(namelist)

      with open(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/extrinsics%02d.txt"%(FLAGS.index), "w") as fo:
        for i in range(3):
          for j in range(3):
            fo.write(str(ref_r[ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t)]) + "\n")

      generateWebGL(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/index.html", w, h, getPlanes(),namelist,sub_sam, f, px, py)
      generateConfigGL(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale+ "/config.js", w, h, getPlanes(),namelist,sub_sam, f, px, py)

def main(argv):
    setGlobalVariables()
    if FLAGS.predict:
        predict()
    else:
        train(3)
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()
