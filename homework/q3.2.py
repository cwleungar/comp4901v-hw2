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
trajectory=np.load('q3_3.npz')['trajectory']
raise NotImplementedError
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the estimated trajectory in blue
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b')
# Plot the ground-truth trajectory in red
#ax.plot(GT_Pose[:, 3], GT_Pose[:, 7], GT_Pose[:, 11], 'r')
# Set the plot title and labels
ax.set_title('Camera Trajectory')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
# Show the plot
plt.show()