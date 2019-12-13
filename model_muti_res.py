## faster render equation
## for multiple resolution
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.compat.v1 import ConfigProto
from mpl_toolkits.mplot3d import Axes3D
import cv2

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
tf.app.flags.DEFINE_integer("epoch", 2000, "Training steps")
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
num_ = 2
nh, nw = 0 , 0
img_ref = 0
is_gen_mpi = True
is_gen_depth = False


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

def network(mpi, depth, rot,tra, is_training):
  alpha, output = 1, 0
  imgs = []
  sublayers = []
  planes = getPlanes()
  print(planes)
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

      a1 =  tf.transpose(depth[i:i+1], perm=[3,1,2,0])
      a2 =  mpia[i:i+1] * a1
      a3 =  mpic[i:i+1] * a2
      img1 = tf.contrib.resampler.resampler(a1,warp)
      img2 = tf.contrib.resampler.resampler(a2,warp)
      img3 = tf.contrib.resampler.resampler(a3,warp)
      weight = tf.cumprod(1 - img1,0,exclusive=True)
      output += tf.reduce_sum(weight*img3,0,keepdims=True)*alpha
      alpha *= 1 - tf.reduce_sum(weight*img2,0,keepdims=True)

  return output[0], alpha

def gen_mpi(mpi,lod_in,is_train=True):
  with tf.variable_scope('gen',reuse=tf.AUTO_REUSE) :
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
        normalizer_params={"is_training": is_train}):
        mpis = []
        for chanel in [32,48]:
          mpis.append(mpi)
          mpi = slim.conv2d(mpi,chanel,[3,3],stride=2) 
        
        mpi = slim.conv2d(mpi,64,[3,3])
        out = slim.conv2d(mpi,3,[3,3],activation_fn=None)
        for i,chanel in enumerate([32,48]):
          mpi = upscale2d(mpi)
          mpi = tf.concat([mpis[-1-i],mpi],-1)#mpis[-1-i] +slim.conv2d(mpi,chanel,[3,3])
          mpi = slim.conv2d(mpi,chanel,[3,3])
          mmm = slim.conv2d(mpi,3,[3,3],activation_fn=None)
          out = upscale2d(out)
          out = lerp_clip(mmm,out,i+lod_in-(2))
        mpi = upscale2d(mpi)
        mmm = slim.conv2d(mpi,3,[3,3],activation_fn=None)
        out = upscale2d(out)
        out = lerp_clip(mmm,out,2+lod_in-(2))
    return out

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
            for chanel in [24,32,24]:
                next = slim.conv2d(next,chanel,[3,3], stride=2)
                next = slim.conv2d(next,chanel,[3,3])
                #next = slim.conv2d(next,chanel,[3,3])

            next = slim.conv2d(next,FLAGS.sublayers,[3,3],
                    biases_initializer=tf.constant_initializer(-5.0),
                    activation_fn=tf.sigmoid)
    return next

def train(logres=4):
    global nh, nw, img_ref
    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    
    features = load_data(0,is_shuff = True)
    rot = features["r"][0]
    tra = features["t"][0]
    real_img = features["img"][0]

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, nh, nw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi, nh, nw, 1]).astype(np.float32)
    int_mpi = np.concatenate([int_mpi1,int_mpi2],-1)
    depth_init = np.random.uniform(-5,0,[num_mpi, int(nh/ FLAGS.subscale), int(nw/ FLAGS.subscale), sub_sam]).astype(np.float32)
    int_noise = np.random.uniform(-1,-1,[num_mpi, int(nh/ 2), int(nw/ 2), 4]).astype(np.float32)

    #lr = 0.1
    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.1)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      if is_gen_mpi:
        noise = tf.compat.v1.get_variable("noise", initializer=int_noise, trainable=True)
        mpic = gen_mpi(noise,lod_in,is_train=True)
        mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=True)
        mpi = tf.concat([mpic,mpia],-1)
      else:
        mpi = tf.compat.v1.get_variable("mpi", initializer=int_mpi, trainable=True)      
      mpi_sig = tf.sigmoid(mpi)
      #mpi = coarse2fine(mpi,lod_in,logres)
      mpi = tf.concat([mpi[:,:,:,:3],coarse2fine(mpi[:,:,:,3:],lod_in,logres)],-1)

      if is_gen_depth:
        if FLAGS.sublayers<1: depth = tf.get_variable("Net_depth", initializer=np.ones((num_mpi,3,3,1)), trainable=False)
        else: depth = gen_depth(mpi,True)
      else:
        depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
        depth = tf.sigmoid(depth)
      depth = tf.image.resize(depth, [nh, nw], align_corners=True)

      
      img_out, allalpha = network(mpi_sig, depth, rot,tra, False)
      long2 = tf.reshape(mpi,(1,num_mpi*nh,nw,4))


    with tf.compat.v1.variable_scope("loss%d"%(FLAGS.index)):
      mask = tf.cast(tf.greater(tf.math.reduce_sum(real_img,2,keepdims=True),0.05),tf.float32)
      fac = (1 - iter/(1500*2))
      tva = tf.constant(0.1) * fac
      tvc = tf.constant(0.005) * 0.001 * fac
      mpiColor = mpi_sig[:, :, :, :3]
      mpiAlpha = mpi_sig[:, :, :, 3:4]
      loss =  100000 * tf.reduce_mean(tf.square(img_out - real_img)*mask)
      loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
      loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
      alpha_loss = 3000 * tf.reduce_mean(allalpha)
      loss += alpha_loss

    t_vars = tf.compat.v1.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'dis' not in var.name]

    netb = tf.contrib.framework.get_variables_by_name("mpi")
    upres = [v.assign(mpi) for i, v in enumerate(netb)]

    image_out = tf.clip_by_value(img_out,0.0,1.0)
    train_op = slim.learning.create_train_op(loss,optimizer,variables_to_train=g_vars)

    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                tf.compat.v1.summary.image("post0/out",tf.expand_dims(tf.concat([real_img,image_out],0),0)),
                tf.compat.v1.summary.image("post1/alpha",long2[:,:,:,3:4]),
                tf.compat.v1.summary.image("post1/color",long2[:,:,:,:3]),
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
    if not FLAGS.restart:
      variables_to_restore = slim.get_variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.latest_checkpoint(localpp )
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(FLAGS.epoch + 3):
        if i<200+50:  nn = 3 - max((i-200)/50,0)
        elif i<400+50:  nn = 2 - max((i-400)/50,0)
        elif i<600+50:  nn = 1 - max((i-600)/50,0)
        else: nn = 0 
        nn += (np.random.rand()*3 - 1.5)
        feedlis = {iter:i,lod_in:nn}

        _,los = sess.run([train_op,loss],feed_dict=feedlis)

        if i%100==0:
            print(FLAGS.index,i, "loss = " ,los)
        if i%20 == 0:
            summ = sess.run(summary,feed_dict=feedlis)
            writer.add_summary(summ,i)
        #if i%50 == 49:
            #sess.run(upres,feed_dict=feedlis)
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
          mpic = gen_mpi2(noise,lod_in,is_train=True)
          mpia = tf.compat.v1.get_variable("mpi_a", initializer=mpi[:,:,:,3:4], trainable=False)
          mpi = tf.concat([mpi,mpia],-1)
        else:
          mpi = tf.compat.v1.get_variable("mpi", initializer=mpi, trainable=False)

        mpi = tf.sigmoid(mpi)
        if is_gen_depth:
          if FLAGS.sublayers<1: depth = tf.get_variable("Net_depth", initializer=np.ones((num_mpi,3,3,1)), trainable=False)
          else: depth0 = gen_depth(mpi,True)
        else:
          depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
          depth0 = tf.sigmoid(depth)
        depth = tf.image.resize(depth0, [nh, nw], align_corners=True)
        img_out,  allalpha = network(mpi, depth, rot,tra, False)

    with tf.compat.v1.variable_scope("post%d"%(FLAGS.index)):
        image_out= tf.clip_by_value(img_out,0.0,1.0)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore()
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
          out = sess.run(image_out,feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
          if(i%50==0): 
            print(i)
            plt.imsave("webpath/"+FLAGS.dataset+"_s%02d"%FLAGS.subscale+"/%04d.png"%( i),out)
          plt.imsave("result/frame/"+FLAGS.dataset+"_s%02d"%FLAGS.subscale+"%04d.png"%( i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p webpath/'+FLAGS.dataset+"_s%02d"%FLAGS.subscale+'/moving.mp4'
        print(cmd)
        os.system(cmd)

    if True:  # make web viewer
      webpath = "webpath/"  #"/var/www/html/orbiter/"
      if not os.path.exists(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale):
          os.system("mkdir " + webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale)

      ret, sublay = sess.run([mpi,depth0])
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
