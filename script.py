import os
import glob
import sys

dataset = 'acup11'
layers = 6
sublayers = 8
codename = 'model_single_withoutbox'
#codename = 'model_orbit_withoutbox'
#codename = 'model_muti_mpi'

if(0):
    for i in range(10):
        print("gen",i)
        os.system("python gen_tfrecord.py -dataset="+ dataset +" -output=tem%d -index=%d -skipexr  "%(i,i))
        #os.system("python gen_tfrecord.py -dataset="+ dataset +" -output=tem%d -index=%d  "%(i,i))

#exit()

for i in range(0,30):
    print("training",i)
    os.system("python "+codename+".py -dataset="+ dataset +" -scale=0.25 -layers=%d -sublayers=%d -index=%d -input=tem%d -epoch=600"%(layers,sublayers,i,i%10))
    print("making",i)
    os.system("python "+codename+".py -dataset="+ dataset +" -scale=0.25 -layers=%d -sublayers=%d -index=%d -input=tem%d -predict"%(layers,sublayers,i,i%10))

os.system("python "+codename+".py -dataset="+ dataset +" -scale=0.25 -layers=%d -sublayers=%d -index=%d -input=tem%d -predict -endvdo"%(layers,sublayers,11,1))
#for i in range(10):
#    print("making",i)
    #os.system("python gen_tfrecord.py -dataset=toro -output=tem"+str(i)+" -index="+str(i)+" -skipexr")
#    os.system("python "+codename+".py -dataset="+ dataset +" -scale=0.22 -layers=%d -sublayers=%d -index=%d -predict -FromFuture"%(layers,sublayers,i))
    #print("training",i)
    #os.system("python model_muti_mpi.py -dataset=temple0 -index="+str(i))
    #print("making",i)
    #os.system("python model_muti_mpi.py -dataset=temple0 -index="+str(i)+" -predict=True")
