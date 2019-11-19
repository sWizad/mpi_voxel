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
  print(index)
  dis=np.linalg.norm(np.transpose(ref_c[index]) - np.array([ccx,ccy,ccz]),2)
  cop = (dx+dy+dz)/3
  bias = -0.1
  dmin = dis-cop#+bias
  dmax = dis+cop+bias
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
          depth_map = depth[i,:,:,j:j+1]
          img = samplePlane(tf.concat([mpi[i], depth_map], -1),rot,tra, dep, 1,index)
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

  output += (1-mask)*bg
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

def train():
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    rot = tf.compat.v1.placeholder(tf.float32, shape=[3,3], name='rotation')
    tra = tf.compat.v1.placeholder(tf.float32, shape=[3,1], name='translation')
    real_img = tf.compat.v1.placeholder(tf.float32, shape=[h,w,3], name='ex_img')

    int_mpi1 = np.random.uniform(-1, 0,[num_mpi, h, w, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-1,[num_mpi, h, w, 1]).astype(np.float32)
    int_mpi = np.concatenate([int_mpi1,int_mpi2],-1)
    depth_init = np.random.uniform(-5,0,[num_mpi, int(h/ FLAGS.subscale), int(w/ FLAGS.subscale), sub_sam]).astype(np.float32)
    #if(FLAGS.sublayers<1):
    #  depth = tf.get_variable("Net_depth", initializer=depth_init*0+1, trainable=False)
    #else:
    depth = tf.get_variable("Net_depth", initializer=depth_init, trainable=True)
    depth0 = tf.sigmoid(depth)

    bg = tf.compat.v1.get_variable("Net_bg", initializer=np.array([1,1,1],dtype=np.float32), trainable=True)
    bg = tf.sigmoid(bg)

    #lr = 0.1
    lr = tf.compat.v1.train.exponential_decay(0.1,lod_in,1000,0.2)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    #optimizer = tf.train.RMSPropOptimizer(lr)
    with tf.compat.v1.variable_scope("Net"):
        mpi = tf.compat.v1.get_variable("mpi", initializer=int_mpi, trainable=True)
        mpi = tf.sigmoid(mpi)
        #depth = gen_depth(mpi,True)
        depth = tf.image.resize(depth0, [h, w], align_corners=True)
        img_out, shifts, sss = network(mpi, depth, bg, rot,tra,(FLAGS.index)%mpi_max, False)
        long = tf.concat(shifts, 1)


    with tf.compat.v1.variable_scope("loss"):
        tva = tf.constant(0.1)
        tvc = tf.constant(0.005) * 0.1
        avl = tf.constant(0.01)
        mpiColor = mpi[:, :, :, :3]
        mpiAlpha = mpi[:, :, :, 3:4]
        loss =  100000 * tf.reduce_mean(tf.square(img_out - real_img))
        loss += tva * tf.reduce_mean(tf.image.total_variation (mpiAlpha))
        loss += tvc * tf.reduce_mean(tf.image.total_variation(mpiColor))
        loss += avl * tf.reduce_mean(tf.image.total_variation(depth0))
        loss += avl * tf.reduce_mean(abs(depth0[:,:,:,:-1] - depth0[:,:,:,1:]) )

        #d_loss = tf.reduce_mean(tf.square(fake_result-1)) + tf.reduce_mean(tf.square(real_result))

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
                        #tf.compat.v1.summary.scalar("dis_loss", d_loss),

                        tf.compat.v1.summary.image("out",tf.expand_dims(image_out,0)),
                        tf.compat.v1.summary.image("origi",tf.expand_dims(real_img,0)),
                        tf.compat.v1.summary.image("alpha",tf.expand_dims(long[:,:,4:5],0)),
                        tf.compat.v1.summary.image("color",tf.expand_dims(long[:,:,:3]*long[:,:,3:4],0)),
                        #tf.compat.v1.summary.image("map_areal",real_result),
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
    #localpp = getLocalPath("/home2/suttisak",'./model/' + FLAGS.dataset +'/'+ FLAGS.input+str(FLAGS.index%mpi_max))
    localpp = './model/' + FLAGS.dataset +'/tem'+str(FLAGS.index%mpi_max)
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
        if FLAGS.index>mpi_max-1:
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

    for i in range(FLAGS.epoch + 3):
        feed = sess.run(features)
        lodin = int(FLAGS.index/mpi_max)*int(FLAGS.epoch/2)
        if i < 6000:
            _,los = sess.run([train_op,loss],feed_dict={lod_in:i,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
        else:
            _,los = sess.run([train_op,loss],feed_dict={lod_in:600,avl:1000,tvc:0.0,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
        
        #for j in range(5):
        #    _,dlos = sess.run([train_dis,d_loss],feed_dict={lod_in:600,rot:feed["r"][0],tra:feed["t"][0],real_img:feed["img"][0]})
    
        if i%100==0:
            print(FLAGS.index,i, "loss = " ,los)
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
        a = 0
        #out, bug = sess.run([img_out,latent])
        #tt = np.concatenate([bug[1,:,:,0],bug[20,:,:,0],bug[40,:,:,0],bug[59,:,:,0]],1)
        #plt.matshow(tt,cmap='gray')
        #plt.show()
    else:
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
        ref_rt = np.array(ref_r[FLAGS.index+ii:FLAGS.index+ii+1])
        ref_tt = np.array(ref_t[FLAGS.index+ii:FLAGS.index+ii+1])
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
            fo.write(str(ref_r[FLAGS.index][ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t[(FLAGS.index)])]) + "\n")

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
