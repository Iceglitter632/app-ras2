import os
import h5py

import shutil

pathname = './train_data2'
files = os.listdir(pathname)

size = len(files)

lefts = []
rigths = []
mids = []

imu_max = [float('-inf') for i in range(10)]
imu_min = [float('inf') for i in range(10)]

speed_max = float('-inf')
speed_min = float('inf')

for ctr, file in enumerate(files):
    print(f'{ctr}/{size} ==> {file}')
    filename = os.path.join(pathname, file)
    h5file = h5py.File(filename, 'r')
 
    f = h5file['others']
#     steer = f[13]
#     if steer > 0:
#         rigths.append(filename)
#     elif steer < 0:
#         lefts.append(filename)
#     else:
#         mids.append(filename)
    for i in range(10):
        if f[i] > imu_max[i]:
            imu_max[i] = f[i]
        if f[i] < imu_min[i]:
            imu_min[i] = f[i]
        
    if f[10] > speed_max:
        speed_max = f[10]
    if f[10] < speed_min:
        speed_min = f[10]
        
    h5file.close()

# print('####################')
# print(f'rights: {len(rigths)}')
# print(f'lefts: {len(lefts)}')        
# print(f'mids: {len(mids)}')

# for idx, f in enumerate(rigths):
#     rounds = 3 if idx%2 else 4
#     for j in range(rounds):
#         shutil.copyfile(f, f[:-3]+f'_{j}.h5')

# for idx, f in enumerate(lefts):
#     for j in range(52):
#         shutil.copyfile(f, f[:-3]+f'_{j}.h5')

imu_max = [str(i) for i in imu_max]
imu_min = [str(i) for i in imu_min]

with open('./values.txt', 'w') as out:
    out.write(','.join(imu_max))
    out.write('\n')
    out.write(','.join(imu_min))
    out.write('\n')
    out.write(str(speed_max)+'\n')
    out.write(str(speed_min))