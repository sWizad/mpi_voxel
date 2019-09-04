import numpy as np
import os
import tensorflow as tf
import glob
import json
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
from utils import findCameraSfm, findExrs

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset", "temple1", "input_dataset")
tf.app.flags.DEFINE_string("output", "tem", "outcome_name")

tf.app.flags.DEFINE_boolean("skipexr", False, "skip_to_convert_exr_file")

tf.app.flags.DEFINE_integer("start", 0, "start")
tf.app.flags.DEFINE_integer("end", 15, "end")

#tf.app.flags.DEFINE_string("ref_img", "0040.png", "reference image such that MPI is perfectly parallel to")
tf.app.flags.DEFINE_integer("index", 1, "index of reference image such that MPI is perfectly parallel to")

#ref_img =  ["0040.png", "0045.png", "0051.png", "0057.png", "0032.png", "0033.png", "0039.png", "0292.png"]
#ref_img = ["354632085", "1221962097", "1004312245", "1902466051", "164864196", "949584407", "496808732", "228538494","354632085"]
#ref_img = ["354632085", "1221962097", "1004312245", "1902466051", "164864196"]

_EPS = np.finfo(float).eps * 4.0
#index = FLAGS.index
ref_t = 0
#cx = cy =0
cx = -0.40782
cy = 3.349134

def readCamera():
  global ref_t
  cams = {}
  path = findCameraSfm(FLAGS.dataset)
  if path == "": return cams

  with open(path, "r") as f:
    js = json.load(f)
  for view in js["views"]:
    #print(view["poseId"])
    #if filter(view["path"]):
      cams[view["poseId"]] = {"filename": view["path"].split("/")[-1][:-3] + "png", "dis":1000, "angle":-1}
      #if ref_img[FLAGS.index] in view["path"]:
        #  for pose in js["poses"]:
        #    if pose["poseId"] == view["poseId"]:
        #      ref_t = np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
        #      break

  #print(cams)
  ref_c = np.transpose(np.array([[0.0,cx,cy]]))
  for pose in js["poses"]:
    #print(pose["poseId"])
    if pose["poseId"] in cams:
      cams[pose["poseId"]]["r"] = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
      cams[pose["poseId"]]["t"] = -cams[pose["poseId"]]["r"] * np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
      cams[pose["poseId"]]["c"] = np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])
      distant = ref_c - cams[pose["poseId"]]["c"]
      #print(ref_c.shape,cams[pose["poseId"]]["c"].shape,distant.shape)
      distant = np.linalg.norm(distant[1:],2)#+0.2*np.linalg.norm(distant[0],2)
      cams[pose["poseId"]]["dis"] = distant
      cams[pose["poseId"]]["angle"] = np.arctan2(cams[pose["poseId"]]["c"][1,0]-cx,cy-cams[pose["poseId"]]["c"][2,0])

  return cams

def convertExr(cams):
  path = "datasets/" + FLAGS.dataset + "/undistorted"
  if not os.path.exists(path):
    os.mkdir(path)

  folder = findExrs(FLAGS.dataset)
  for f in glob.glob(folder + "/*.exr"):
    imgname = f.split("/")[-1][:-4]
    if imgname not in cams or os.path.exists(path + "/" + cams[imgname]["filename"]): continue
    os.system("convert " + f + " " + path + "/" + cams[imgname]["filename"][:-3] + "png")
    print(cams[imgname]["filename"])
  return path

def quaternion_from_matrix(matrix):
    matrix = np.r_[np.c_[matrix, np.array([[0],[0],[0]])], np.array([[0,0,0,1]])]
    # Return quaternion from rotation matrix.
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def quaternion_matrix(quaternion):
    # Return homogeneous rotation matrix from quaternion.
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def unit_vector(data, axis=None, out=None):
    # Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    data = np.array(data, dtype=np.float64, copy=True)
    data /= math.sqrt(np.dot(data, data))
    return data

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    # Return spherical linear interpolation between two quaternions.
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def interpolate_rotation(m1, m2, t):
  q1 = quaternion_from_matrix(m1)
  q2 = quaternion_from_matrix(m2)
  return quaternion_matrix(quaternion_slerp(q1, q2, t))[:3, :3]

def generate():
    import imagesize

    def f1(a):
        return tf.train.Feature(float_list=tf.train.FloatList(value=np.array(a).flatten())),
    def bytes_feature(values):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
    def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    cams = readCamera()
    scams = sorted(cams.items(), key=lambda x: x[1]["filename"])
    #dcams = sorted(cams.items(), key=lambda x: x[1]["angle"])
    imgFolder = "datasets/" + FLAGS.dataset + "/undistorted" if FLAGS.skipexr else convertExr(cams)
    with tf.io.TFRecordWriter("datasets/" + FLAGS.dataset + "/" + FLAGS.output +".train") as tfrecord_writer:
      with tf.Session() as sess:
        px, py = [], []
        for i in range(25):
            print(i,scams[i][1]["filename"])
            #if dcams[i][1]["angle"]>cams[ref_img[index+1]]["angle"]:
            #    break
            #if dcams[i][1]["angle"]>cams[ref_img[index]]["angle"] and dcams[i][1]["dis"]>4.0:
            if(1):
                try:
                    image_path = imgFolder + "/" + scams[i][1]["filename"]
                    width, height = imagesize.get(image_path)
                    image = tf.convert_to_tensor(image_path, dtype = tf.string)
                    image = tf.io.read_file(image)
                    ret = sess.run(image)
                    example = tf.train.Example(features=tf.train.Features(
                      feature={
                        'img' : bytes_feature(ret),
                        'r': f1(scams[i][1]["r"]),
                        't': f1(scams[i][1]["t"]),
                        'h': _int64_feature(height),
                        'w': _int64_feature(width),
                      }))
                    tfrecord_writer.write(example.SerializeToString())
                    #px.append(dcams[i][1]["c"][1])
                    #py.append(dcams[i][1]["c"][2])
                except:
                    print("file not exist")

    #plt.scatter(px,py)
    #plt.show()
    #scams = sorted(dcams[:30], key=lambda x: x[1]["filename"])
    with tf.io.TFRecordWriter("datasets/" + FLAGS.dataset + "/" + FLAGS.output + ".test") as tfrecord_writer:
        dirt = [15, 19, 3, 0]
        n = 100
        m = len(dirt)-1
        #print(scams[FLAGS.start][1]["filename"])
        #print(scams[FLAGS.end][1]["filename"])
        for i in range(m):
            print(scams[dirt[i]][1]["filename"]," to" ,scams[dirt[i+1]][1]["filename"])

            for j in range(n):
                  tt = (j+0.5) / n # avoid the train image, especially the ref image
                  rot = interpolate_rotation(scams[dirt[i]][1]["r"], scams[dirt[i+1]][1]["r"], tt)
                  t = scams[dirt[i]][1]["t"] * (1-tt) + scams[dirt[i+1]][1]["t"] * tt
                  #rot = interpolate_rotation(cams[ref_img[index]]["r"], cams[ref_img[index+1]]["r"], tt)
                  #t = cams[ref_img[index]]["t"] * (1-tt) + cams[ref_img[index+1]]["t"] * tt
                  example = tf.train.Example(features=tf.train.Features(
                    feature={
                      'r': f1(rot),
                      't': f1(t),
                    }))
                  tfrecord_writer.write(example.SerializeToString())
                  #print(scams[i])


if __name__ == "__main__":
    """
    print("Hey!!")
    cams = readCamera()
    dcams = sorted(cams.items(), key=lambda x: x[1]["dis"])
    px, py,area = [], [], []
    for i in range(300):
    #    cx += dcams[i][1]["c"][1]
    #    cy += dcams[i][1]["c"][2]
        if dcams[i][1]["dis"]>4 and dcams[i][1]["angle"]<cams[ref_img[1]]["angle"]:
            print(dcams[i][1]["dis"])
            area.append((dcams[i][1]["angle"]/np.pi+1)*100)
            px.append(dcams[i][1]["c"][1])
            py.append(dcams[i][1]["c"][2])
    #cx /= 300
    #cy /= 300
    plt.scatter(px,py,s=area)
    #plt.scatter([cx],[cy])
    """
    #cams = readCamera()
    #px, py= [],[]
    #for id in ref_img:
        #print(cams[id]["c"].shape)
    #    px.append(cams[id]["c"][1,0])
    #    py.append(cams[id]["c"][2,0])

    #plt.scatter(px,py,c='r')
    #plt.show()
    #dcams = sorted(cams.items(), key=lambda x: x[1]["dis"])
    #scams = sorted(dcams[:30], key=lambda x: x[1]["filename"])
    generate()
