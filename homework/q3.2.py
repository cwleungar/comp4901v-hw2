from submission import *
import matplotlib.pyplot as plt
file=open('../data/GTPoses.npz','rb')
file=file.readlines()
GTPoses=[]
for i in file:
    GTPoses.append(list(i.split()))
GTPoses=np.array(GTPoses,dtype=float)
t=visualOdometry('../data/monocular video sequence/data',GT_Pose=GTPoses)
np.savez('q3_3.npz',trajectory=t)
raise NotImplementedError
trajectory=np.load('q3_3.npz')['trajectory']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b')
GT=np.stack((GTPoses[:, 3], GTPoses[:, 7], GTPoses[:, 11]),axis=1)
print(trajectory)
print(GT)
#ax.plot(GTPoses[:, 3], GTPoses[:, 7], GTPoses[:, 11], 'r')
ax.set_title('Camera Trajectory')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
# Show the plot
plt.show()