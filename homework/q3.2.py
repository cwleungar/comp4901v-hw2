from submission import *

file=open('../data/GTPoses.npz','rb')
file=file.readlines()
GTPoses=[]
for i in file:
    GTPoses.append(list(i.split()))
GTPoses=np.array(GTPoses,dtype=float)
visualOdometry('../data/monocular video sequence/data',GT_Pose=GTPoses)