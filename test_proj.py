import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
from utils import *

dataset = 'acup7'
ref_img = ["0_00000","0_00004","0_00008","0_00012","0_00016","0_00020","0_00024","0_00028","0_00032","0_00036","0_00000"]

index = 1
ref_r = []
ref_t = []
ref_c = []
mpi_max = 10
f = px = py = 0
dmin, dmax = -1, -1
num_mpi = 5
scale = 0.2
offset = 0

def load_data(i,is_shuff=False):
  global dataset
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
    if scale < 1:
      fs["img"] = tf.image.resize_images(fs["img"], [h, w], tf.image.ResizeMethod.AREA)

    fs["r"] = tf.reshape(fs["r"], [3, 3])
    fs["t"] = tf.reshape(fs["t"], [3, 1])
    return fs

  # np.random.shuffle(filenames)
  localpp = "datasets/" + dataset + "/tem" + str(index%mpi_max) + ".train"
  dataset = tf.data.TFRecordDataset([localpp])
  dataset = dataset.map(parser)
  if(is_shuff):  dataset = dataset.shuffle(5)
  dataset = dataset.repeat().batch(1)

  return dataset.make_one_shot_iterator().get_next()

def setGlobalVariables():
  global f, px, py, ref_r, ref_t, ref_c, w, h

  path = findCameraSfm(dataset)
  with open(path, "r") as f:
    js = json.load(f)


  f = float(js["intrinsics"][0]["pxFocalLength"]) * scale
  px = float(js["intrinsics"][0]["principalPoint"][0]) * scale
  py = float(js["intrinsics"][0]["principalPoint"][1]) * scale
  w = int(int(js["intrinsics"][0]["width"]) * scale) 
  h = int(int(js["intrinsics"][0]["height"]) * scale)

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
     with open("datasets/" + dataset + "/planes.txt", "r") as fi:
       ccx,dx = [float(x) for x in fi.readline().split(" ")]
       ccy,dy = [float(x) for x in fi.readline().split(" ")]
       ccz,dz = [float(x) for x in fi.readline().split(" ")]
     #ccy += 0.30
     #ccz += 0.1
     #ccx -= 0.05

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

def sampleDepth(latent,d,index):
    global f, px, py, ref_r, ref_t, w, h
    x, y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    wan = np.ones_like(x)

    #new_r = np.matmul(ref_r[index], np.transpose(ref_r[index]))
    #new_t = np.matmul(np.matmul(-ref_r[index], np.transpose(ref_r[index])), ref_t[index]) + ref_t[index]

    coords = np.stack([x*d, y*d, wan*d, wan], -1).reshape(h,w,4,1) #(h,w,4)
    k = [[2*f/w, 0      , 2*px/w-1  , 0],
         [0    , -2*f/h , -2*py/h+1 , 0],
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
    return p1#*tf.expand_dims(b,-1)

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
  dmin = dis-cop
  dmax = dis+cop
  return np.linspace(dmin, dmax, num_mpi)

def test():
    features = load_data(0,is_shuff = True)
    latent = np.ones([60, 60, 60, 1],dtype=np.float32)
    latent = tf.get_variable("Net_depth", initializer=latent, trainable=False)
    img = features['img'][0]
    rot = features['r'][0]
    tra = features['t'][0]
    planes = getPlanes(index)
    mask = 1.0
    for i in range(5):
        plane = planes[i]
        depal = sampleDepth(latent,plane,index)
        depal = samplePlane(depal, rot, tra, plane, ks=1,index=0)
        depal = depal[0]
        ii = i/5
        color = np.array([ii,0.2,1-ii]).reshape((1,1,3))
        img = (img*(1-depal) + color*img*depal)*(mask) + img*(1-mask)
        mask *= (1-depal)




    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #feed = sess.run(features)
    sess.run(tf.global_variables_initializer())
    #ddd = sess.run(depal)
    #print(ddd)
    #ddd = np.tile(ddd,(1,1,3))
    out = sess.run(img)
    plt.imshow(out)
    plt.show()

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  setGlobalVariables()
  test()
  #tf.compat.v1.app.run()