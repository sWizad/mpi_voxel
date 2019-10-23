import os
import glob
import sys

dataset = 'toro2'

for i in range(10):
    print("gen",i)
    os.system("python gen_tfrecord.py -dataset="+ dataset +" -output=tem%d -index=%d -skipexr  "%(i,i))

#exit()
#for i in range(20,30):
#    print("training",i)
#    os.system("python model_muti_mpi.py -dataset="+ dataset +" -scale=0.2 -index="+str(i)+" -epoch=400")
#    #print("making",i)
#    os.system("python model_muti_mpi.py -dataset="+ dataset +" -scale=0.2 -index="+str(i)+" -predict=True")

#for i in range(10):
#    print("making",i)
    #os.system("python gen_tfrecord.py -dataset=toro -output=tem"+str(i)+" -index="+str(i)+" -skipexr")
#    os.system("python model_muti_mpi.py -dataset="+ dataset +" -scale=0.2 -index="+str(i)+" -predict -FromFuture")
    #print("training",i)
    #os.system("python model_muti_mpi.py -dataset=temple0 -index="+str(i))
    #print("making",i)
    #os.system("python model_muti_mpi.py -dataset=temple0 -index="+str(i)+" -predict=True")
