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
tf.app.flags.DEFINE_boolean("FromFuture", False, "making a video")
tf.app.flags.DEFINE_float("scale", 0.75, "scale input image by")
tf.app.flags.DEFINE_integer("subscale", 8, "downscale factor for the sub layer")

tf.app.flags.DEFINE_integer("layers", 25, "number of planes")
tf.app.flags.DEFINE_integer("sublayers", 2, "number of sub planes")
tf.app.flags.DEFINE_integer("epoch", 1000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_integer("index", 0, "index number")

tf.app.flags.DEFINE_string("dataset", "temple0", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "tem0", "input tfrecord")

#tf.app.flags.DEFINE_string("ref_img", "0051.png", "reference image such that MPI is perfectly parallel to")
if FLAGS.dataset == "temple0":
    ref_img = ["0040.png", "0045.png", "0051.png", "0057.png", "0032.png", "0033.png", "0039.png", "0292.png", "0040.png"]
    ref_ID = ["354632085", "1221962097", "1004312245", "1902466051", "164864196", "949584407", "496808732", "228538494","354632085"]
elif FLAGS.dataset == "lib2":
    ref_img = ["2_00000","2_00004","2_00008","2_00012","2_00016","2_00000"]
elif FLAGS.dataset == "toro" :
    ref_img = ["2_00000","2_00004","2_00008","2_00012","2_00016","2_00020","2_00024","2_00028","2_00032","2_00036","2_00000"]
elif FLAGS.dataset in ["acup11"] :
    ref_img = ["4_00000","4_00004","4_00008","4_00012","4_00016","4_00020","4_00024","4_00028","4_00032","4_00036","4_00000"]
elif FLAGS.dataset in ["acup7", "acup6","toro2","dumdum"]:
    ref_img = ["0_00000","0_00004","0_00008","0_00012","0_00016","0_00020","0_00024","0_00028","0_00032","0_00036","0_00000"]
else:
    print("error: dataset not found")
    exit()
mpi_max = len(ref_img)-1

laten_h, laten_w, laten_d, laten_ch = 60,60,60,4#50, 50, 50, 4
#laten_h, laten_w, laten_d, laten_ch = 35,20,20,4
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
      fs["img"] = tf.image.resize_images(fs["img"], [h, w], tf.image.ResizeMethod.AREA)

    fs["r"] = tf.reshape(fs["r"], [3, 3])
    fs["t"] = tf.reshape(fs["t"], [3, 1])
    return fs

  # np.random.shuffle(filenames)
  #localpp = getLocalPath("/home2/suttisak","datasets/" + FLAGS.dataset + "/tem" + str(FLAGS.index%mpi_max) + ".train")
  localpp = "datasets/" + FLAGS.dataset + "/tem" + str(FLAGS.index%mpi_max) + ".train"
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

def sampleDepth(latent,d,index):
    global f, px, py, ref_r, ref_t, w, h
    x, y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    wan = np.ones_like(x)

    coords = np.stack([x*d, y*d, wan*d, wan], -1).reshape(h,w,4,1) #(h,w,4)
    k = [[2*f/w, 0      , 2*px/w-1  , 0],
         [0    , 2*f/h ,  2*py/h-1 , 0],
         [0    , 0      , 1         , 0],
         [0    , 0      , 0         , 1]]
    ki = np.linalg.inv(k)
    mv = np.zeros([4,4])
    mv[:3,:3] = ref_r[index]
    for i in range(3): mv[i,3] = ref_t[index][i]
    mv[3,3] = 1
    mvi= np.linalg.inv(mv)
    #cam coordinate to world coordinate
    ncoords = np.matmul(mvi,np.matmul(ki,coords))[:,:,:,0]
    
    #world coordinate to box coordinate
    cx = (ncoords[:, :, 0]-ccx+dx)/dx/2
    cy = (ncoords[:, :, 1]-ccy+dy)/dy/2
    cz = (ncoords[:, :, 2]-ccz+dz)/dz/2
    
    p1 = linear3d_sampler(latent, cx, cy, cz)
    tf.add_to_collection("checkpoints", p1)
    b =tf.cast(tf.greater(cx,-0.01),tf.float32)*tf.cast(tf.greater(cy,-0.01),tf.float32)*tf.cast(tf.greater(cz,-0.01),tf.float32)
    b *= tf.cast(tf.greater(1.01,cx),tf.float32)*tf.cast(tf.greater(1.01,cy),tf.float32)*tf.cast(tf.greater(1.01,cz),tf.float32)
    return p1*tf.expand_dims(b,-1)

def getPlanes(index):
  dis=np.linalg.norm(np.transpose(ref_c[index]) - np.array([ccx,ccy,ccz]),2)
  cop = (dx+dy+dz)/3
  dmin = dis-cop
  dmax = dis+cop
  if FLAGS.invz:
    return 1/np.linspace(1, 0.0001, num_mpi) * dmin
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
      aa = 1
      out = 0
      for j in range(sub_sam):
          vv = j/sub_sam
          dep = rplanes[i]*(1-vv) + rplanes[i+1]*(vv)
          #depth = sampleDepth(latent,dep, index)
          depth_map = depth[i,:,:,j:j+1]
          img = samplePlane(tf.concat([mpi[i], depth_map], -1),rot,tra, dep, 1,index)
          tf.add_to_collection("checkpoints", img)
          img = img[0]
          out += img[:,:,:4]*img[:,:,4:5]*aa
          aa  *= (1-img[:,:,4:5])
          depth_map = tf.image.resize_images(depth_map, [int(h/8), int(w/8)], tf.image.ResizeMethod.AREA)
          sublayers.append(depth_map)
          if j == 0:
              imgs.append(img)
      output += out[:,:,:3]*out[:,:,3:4]*alpha
      alpha *= (1-out[:,:,3:4])
      mask += out[:,:,3:4]*alpha

  output += (1-mask)*bg
  return output, imgs, sublayers

def train():
    lod_in = tf.placeholder(tf.float32, shape=[], name='lod_in')
    rot = tf.placeholder(tf.float32, shape=[3,3], name='rotation')
    tra = tf.placeholder(tf.float32, shape=[3,1], name='translation')
    real_img = tf.placeholder(tf.float32, shape=[h,w,3], name='ex_img')

    int_mpi1 = np.random.uniform(-1, 0,[num_mpi, h, w, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-1,[num_mpi, h, w, 1]).astype(np.float32)
    int_mpi = np.concatenate([int_mpi1,int_mpi2],-1)


    depth_init = np.random.uniform(-5,0,[num_mpi, int(h/ FLAGS.subscale), int(w/ FLAGS.subscale), sub_sam]).astype(np.float32)
    #depth_init = np.random.uniform(-5, 0, [FLAGS.layers, int(nh / FLAGS.subscale), int(nw / FLAGS.subscale), FLAGS.sublayers]).astype(np.float32)

    #backup = tf.get_variable("Net_backup", initializer=depth_init, trainable=False)
    if(FLAGS.sublayers<=1):
      depth = tf.get_variable("Net_depth", initializer=depth_init*0+1, trainable=False)
    else:
      depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
    depth = tf.sigmoid(depth)
    depth = tf.image.resize(depth, [h, w], align_corners=True)

    bg = tf.get_variable("Net_bg", initializer=np.array([1,1,1],dtype=np.float32), trainable=True)
    bgup = tf.get_variable("Net_backupbg", initializer=np.array([1,1,1],dtype=np.float32), trainable=False)
    #noise = 0.15*(2*tf.random_uniform(bg.shape)-1) * (1-lod_in/2200)
    bg = tf.sigmoid(bg)

    mpis = []
    image_out = []
    longs = []
    losses = []
    summaries = []
    train_ops = []

    #lr = 0.1
    lr = tf.compat.v1.train.exponential_decay(0.1,lod_in,1000,0.1)
    optimizer = tf.train.AdamOptimizer(lr)
    #optimizer = tf.train.RMSPropOptimizer(lr)
    for i in range(num_):
        with tf.compat.v1.variable_scope("Net%d"%((FLAGS.index+i)%mpi_max)):
            mpi = tf.get_variable("mpi", initializer=int_mpi, trainable=True)
            mpi = tf.sigmoid(mpi)
            mpis.append(mpi)
            img_out, shifts, sss = network(mpis[i], depth, bg, rot,tra,(FLAGS.index+i)%mpi_max, False)
            long = tf.concat(shifts, 1)

        with tf.compat.v1.variable_scope("loss%d"%((FLAGS.index+i)%mpi_max)):
            tva = tf.constant(0.1)
            tvc = tf.constant(0.005)
            mpiColor = mpi[:, :, :, :3]
            mpiAlpha = mpi[:, :, :, 3:4]
            loss =  100000 * tf.reduce_mean(tf.square(img_out - real_img))
            loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
            loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
            losses.append(loss)

        with tf.compat.v1.variable_scope("post%d"%((FLAGS.index+i)%mpi_max)):
            image_out.append(tf.clip_by_value(img_out,0.0,1.0))
            longs.append(tf.clip_by_value(long,0.0,1.0) )
            train_ops.append(slim.learning.create_train_op(losses[i],optimizer))

            summary = tf.compat.v1.summary.merge([
                            tf.compat.v1.summary.scalar("all_loss", losses[i]),
                            tf.compat.v1.summary.image("out",tf.expand_dims(image_out[i],0)),
                            tf.compat.v1.summary.image("origi",tf.expand_dims(real_img,0)),
                            tf.compat.v1.summary.image("alpha",tf.expand_dims(longs[i][:,:,3:4],0)),
                            tf.compat.v1.summary.image("color",tf.expand_dims(longs[i][:,:,:3]*longs[i][:,:,3:4],0)),
                            ])
            summaries.append(summary)

    #netb = tf.contrib.framework.get_variables_by_name("Net_backup")
    #netb.append(tf.contrib.framework.get_variables_by_name("Net_backupbg")[0])
    #netd = tf.contrib.framework.get_variables_by_name("Net_depth")
    #netd.append(tf.contrib.framework.get_variables_by_name("Net_bg")[0])
    #backup = [v.assign(netd[i]) for i, v in enumerate(netb)]
    #loadup = [v.assign(netd[i]*0.5+netb[i]*0.5) for i, v in enumerate(netd)]
    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    #localpp = getLocalPath("/home2/suttisak","TensorB/"+FLAGS.dataset)
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
    localpp = './model/' + FLAGS.dataset +'/'+ FLAGS.input+str(FLAGS.index%mpi_max)
    if not os.path.exists(localpp):
        os.makedirs(localpp)

    for i_mpi in range(num_):
        features0 = load_data(i_mpi,is_shuff = False)
        features = load_data(i_mpi,is_shuff = True)
        sess = tf.Session(config=config)
        if i_mpi == 0:
            sess.run(tf.global_variables_initializer())
            if FLAGS.index>0:
                t_vars = slim.get_variables_to_restore()
                vars_to_restore = [var for var in t_vars if 'Net%d'%((FLAGS.index+1)%mpi_max) not in var.name and 'Net' in var.name and 'Adam' not in var.name]
                saver = tf.train.Saver(vars_to_restore)
                #localpp0 = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index-1)%mpi_max))
                localpp0 = './model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index-1)%mpi_max)
                ckpt = tf.train.latest_checkpoint(localpp0)
                saver.restore(sess, ckpt)
                saver = tf.train.Saver()
        else:
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            ckpt = tf.train.latest_checkpoint(localpp )
            saver.restore(sess, ckpt)

        feed0 = sess.run(features0)
        if (FLAGS.index==0 and i_mpi==0): FLAGS.epoch = int(FLAGS.epoch*1.5)
        if (FLAGS.index==0 and i_mpi==1): FLAGS.epoch = int(FLAGS.epoch/1.5)
        for i in range(FLAGS.epoch + 3):

            feed = sess.run(features)
            lodin = int(FLAGS.index/mpi_max)*int(FLAGS.epoch/2)
            if (i_mpi==0 and i<2500):
                _,los = sess.run([train_ops[i_mpi],losses[i_mpi]],feed_dict={lod_in:lodin+i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
            elif(i_mpi==0 and FLAGS.epoch - i < 0):
                _,los = sess.run([train_ops[i_mpi],losses[i_mpi]],feed_dict={lod_in:lodin+i,tva:0.00001,rot:feed0["r"][0],tra:feed0["t"][0],real_img:feed0["img"][0]})
            else:
                _,los = sess.run([train_ops[i_mpi],losses[i_mpi]],feed_dict={lod_in:lodin+i,tva:0.001,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})

            if i%100==0:
                print(FLAGS.index,i_mpi,i, "loss = " ,los)
            if i%20 == 0:
               #if(FLAGS.index>mpi_max-8): sess.run(loadup)
               if(FLAGS.epoch - i < 20):
                 summ = sess.run(summaries[i_mpi],feed_dict={lod_in:lodin+i,rot:feed0["r"][0],tra:feed0["t"][0],real_img:feed0["img"][0]})
               else:
                 summ = sess.run(summaries[i_mpi],feed_dict={lod_in:lodin+i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
               blank = 0.0
               if(i_mpi==0 and FLAGS.index!=0): blank += FLAGS.epoch
               writer.add_summary(summ,blank+i)
            if i%200==1:
               #sess.run(backup)
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

    lod_in = tf.placeholder(tf.float32, shape=[], name='lod_in')
    rot = tf.placeholder(tf.float32, shape=[3,3], name='rotation')
    tra = tf.placeholder(tf.float32, shape=[3,1], name='translation')


    testset = tf.data.TFRecordDataset(["datasets/" + FLAGS.dataset + "/tem" + str(FLAGS.index%mpi_max)+ ".test"])
    testset = testset.map(parser).repeat().batch(1).make_one_shot_iterator()
    features = testset.get_next()


    bg = tf.get_variable("Net_bg", initializer=np.array([1,1,1],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)

    depth_init = np.random.uniform(-5,0,[num_mpi, int(h/ FLAGS.subscale), int(w/ FLAGS.subscale), sub_sam]).astype(np.float32)

    mpi = np.zeros([num_mpi, h, w, 4],dtype=np.float32)
    mpi[0] = [1.,0.,0.,.95]
    mpi[1] = [1.,.5,0.,.95]
    mpi[2] = [1.,1.,0.,.95]
    mpi[3] = [.5,1.,0.,.95]

    depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
    depth = tf.sigmoid(depth)
    depth = tf.image.resize(depth, [h, w], align_corners=True)

    mpis = []
    image_out = []
    longs = []
    ssss = []

    for i in range(num_):
        with tf.compat.v1.variable_scope("Net%d"%((FLAGS.index+i)%mpi_max)):
            mpi = tf.get_variable("mpi", initializer=mpi, trainable=True)
            mpi = tf.sigmoid(mpi)
            mpis.append(mpi)

            img_out, shifts, sss = network(mpis[i], depth, bg, rot,tra,(FLAGS.index+i)%mpi_max, False)
            ssss.append(sss)
            long = tf.concat(shifts, 1)

        with tf.compat.v1.variable_scope("post%d"%((FLAGS.index+i)%mpi_max)):
            image_out.append(tf.clip_by_value(img_out,0.0,1.0))
            longs.append(tf.clip_by_value(long,0.0,1.0) )

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #config = ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    #localpp = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index)%mpi_max))
    localpp = './model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index)%mpi_max)
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)

    if FLAGS.FromFuture:
        t_vars = slim.get_variables_to_restore()
        vars_to_restore = [var for var in t_vars if 'Net%d'%((FLAGS.index+1)%mpi_max) in var.name and 'Adam' not in var.name]
        #print(vars_to_restore)
        saver = tf.train.Saver(vars_to_restore)
        #localpp0 = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index+1)%mpi_max))
        localpp0 = './model/' + FLAGS.dataset +'/'+ FLAGS.input+str((FLAGS.index+1)%mpi_max)
        ckpt = tf.train.latest_checkpoint(localpp0 )
        saver.restore(sess, ckpt)


    if False:
        a = 0
        #out, bug = sess.run([img_out,latent])
        #tt = np.concatenate([bug[1,:,:,0],bug[20,:,:,0],bug[40,:,:,0],bug[59,:,:,0]],1)
        #plt.matshow(tt,cmap='gray')
        #plt.show()
    else:
        for i in range(0,300,1):
            feed = sess.run(features)
            if(i%50==0): print(i)
            #out0, out1 = sess.run([image_out[0],image_out[1]])
            out0 = sess.run(image_out[0],feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
            out1 = sess.run(image_out[1],feed_dict={rot:feed["r"][0],tra:feed["t"][0]})
            #out, bug = sess.run([img_out,long])
            #out0 = np.rot90(out0)
            #out1 = np.rot90(out1)
            out = (1-i/300)*out0 + (i/300)*out1 #np.concatenate((out0,out1),1)
            #plt.imsave("result/%04d.png"%(i),bug[:,:,:3])
            plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%(300*(FLAGS.index%mpi_max) + i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p result/'+FLAGS.dataset+'_moving.mp4'
        print(cmd)
        os.system(cmd)

    if False:
      webpath = "/var/www/html/orbiter/"
      if not os.path.exists(webpath + FLAGS.dataset):
          os.system("mkdir " + webpath + FLAGS.dataset)


      for ii in range(1):
        ref_rt = np.array(ref_r[FLAGS.index+ii:FLAGS.index+ii+1])
        ref_tt = np.array(ref_t[FLAGS.index+ii:FLAGS.index+ii+1])
        print(ref_rt.shape)
        ret, sublay = sess.run([mpis[ii],ssss[ii]],feed_dict={features['r']:ref_rt,features['t']:ref_tt})
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
            fo.write(str(ref_r[FLAGS.index][ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t[(FLAGS.index)])]) + "\n")

      generateWebGL(webpath + FLAGS.dataset+ "/index.html", w, h, getPlanes(),namelist,sub_sam, f, px, py)

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
