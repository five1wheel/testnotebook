```
/home/zc/anaconda3/envs/myenv/bin/python3.8 /snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py --mode=client --host=127.0.0.1 --port=33637 
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/zc/Project/ContextPose-PyTorch-release-master'])
PyDev console: starting.
Python 3.8.20 (default, Oct  3 2024, 15:24:27) 
[GCC 11.2.0] on linux
pred_results = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/train.pkl', allow_pickle=True)
keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
NameError: name 'np' is not defined
import numpy as np 
pred_results = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/train.pkl', allow_pickle=True)
keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
gt_results = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', allow_pickle=True)
gt_results = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', allow_pickle=True).item()
pred_results_val = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/val.pkl', allow_pickle=True)
gt_results['table'][0][0][0]
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
IndexError: invalid index to scalar variable.
gt_results['table'][0]
(0, 0, 0, [[-261.84055 ,  186.55287 ,   61.438904], [-188.47029 ,   14.077108,  475.1688  ], [-223.23566 ,  163.80551 ,  890.5342  ], [  39.877888,  145.00249 ,  923.9878  ], [ -11.675992,  160.8992  ,  484.39148 ], [ -51.55029 ,  220.14626 ,   35.834385], [ -91.679   ,  154.404   ,  907.261   ], [-132.34781 ,  215.7302  , 1128.8396  ], [ -97.167404,  202.34435 , 1383.1466  ], [-120.032906,  190.96475 , 1573.3999  ], [-350.77136 ,   43.44217 ,  831.3473  ], [-315.40536 ,  164.55286 , 1049.1747  ], [-230.36957 ,  203.17921 , 1311.9639  ], [  25.895449,  192.35948 , 1296.1571  ], [ 107.10583 ,  116.0503  , 1040.5062  ], [ 129.83815 ,  -48.024902,  850.94806 ], [-112.97075 ,  127.969444, 1477.4457  ]], [[264, 264, 658, 658], [235, 351, 612, 728], [196, 442, 581, 827], [136, 272, 635, 771]])
gt_results['table'][0][0]
0
gt_results['table'][0][0,0,0]
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
IndexError: too many indices for array: array is 0-dimensional, but 3 were indexed
s = []
for t in gt_results['table']:
    if t[0]==0:
        if t[1]==0:
            s.append(t)    
            
ss = []
for t in gt_results['table']:
    if t[0]==0:
        if t[1]==2:
            ss.append(t)    
            
sss = []
for t in gt_results['table']:
    if t[0]==1:
        if t[1]==0:
            sss.append(t)    
            
sss2 = []
for t in gt_results['table']:
    if t[0]==1:
        if t[1]==0:
            sss2.append(t)    
            
sss2 = []
for t in gt_results['table']:
    if t[0]==1:
        if t[1]==2:
            sss2.append(t)    
            
            
sss = []
for t in gt_results['table']:
    if t[0]==5:
        if t[1]==0:
            sss.append(t)  
            
sss2 = []
for t in gt_results['table']:
    if t[0]==5:
        if t[1]==2:
            sss2.append(t)   
            
for t in gt_results['table']:
    if t[0]==5:
        if t[1]==1:
            sss2.append(t)   
            
for t in gt_results['table']:
    if t[0]==0:
        if t[1]==1:
            sss2.append(t)   
            
data = pred_results.copy()
aa = data['keypoints_3d'][0:500]+data['keypoints_3d'][1115:2193]
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (500,17,3) (1078,17,3) 
aa = np.concatenate(data['keypoints_3d'][0:500],data['keypoints_3d'][1115:2193])
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
  File "<__array_function__ internals>", line 200, in concatenate
TypeError: only integer scalar arrays can be converted to a scalar index
aa = np.concatenate((data['keypoints_3d'][0:500],data['keypoints_3d'][1115:2193]),axis=0)
bb = np.concatenate((pred_results_val['keypoints_3d'][0:37],pred_results_val['keypoints_3d'][172:264]),axis=0)
cc = np.concatenate((data['indexes'][0:500],data['indexes'][1115:2193]),axis=0)
sdw=  data[75881]
sdw=  data['table'][75881]
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
IndexError: list index out of range
sdw = gt_results['table'][75881]
sdw = gt_results['table'][75881].item()
sdw1 = aa[0]
kp1 = sdw[3]
kp2 = sdw1
CONNECTIVITY_DICT = {"human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8),
                                  (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]}
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test= kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax1.plot(plotX_g, plotY_g, plotZ_g)
plt.show()
Backend TkAgg is interactive backend. Turning interactive mode on.
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 13, in <module>
NameError: name 'ax1' is not defined
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test= kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
plt.show()
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test= kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
ax1.set_xlim3d(-1500, 1500)
ax1.set_ylim3d(-1500, 1500)
ax1.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
    plotX_p = [keypoints_3d_gt[i][0] for i in group]
    plotY_p = [keypoints_3d_gt[i][1] for i in group]
    plotZ_p = [keypoints_3d_gt[i][2] for i in group]
    ax1.plot(plotX_p, plotY_p, plotZ_p)
plt.show()
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test= kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
ax1.set_xlim3d(-1500, 1500)
ax1.set_ylim3d(-1500, 1500)
ax1.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
    plotX_p = [keypoints_3d_gt[i][0] for i in group]
    plotY_p = [keypoints_3d_gt[i][1] for i in group]
    plotZ_p = [keypoints_3d_gt[i][2] for i in group]
    ax1.plot(plotX_p, plotY_p, plotZ_p)
plt.show()
data = pred_results.copy()
data_2 = pred_results_val.copy()
data_3 = pred_results_val.copy()
data['keypoints_3d'] =  aa
data['indexes'] = cc 
data_2['keypoints_3d'] =  bb
data_2['indexes'] = np.concatenate((pred_results_val['indexes'][0:37],pred_results_val['indexes'][172:264]),axis=0)
import pickle
with open('train_1.pkl','wb') as f:
    pickle.dump(data,f)
with open('val_1.pkl', 'wb') as f:
    pickle.dump(data_2, f)
    
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Image.__del__ at 0x7ff5213ee5e0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Image.__del__ at 0x7ff5213ee5e0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Image.__del__ at 0x7ff5213ee5e0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
agg = gt_results['table'].copy()
cgg  = agg.copy()
cgg[0]
(0, 0, 0, [[-261.84055 ,  186.55287 ,   61.438904], [-188.47029 ,   14.077108,  475.1688  ], [-223.23566 ,  163.80551 ,  890.5342  ], [  39.877888,  145.00249 ,  923.9878  ], [ -11.675992,  160.8992  ,  484.39148 ], [ -51.55029 ,  220.14626 ,   35.834385], [ -91.679   ,  154.404   ,  907.261   ], [-132.34781 ,  215.7302  , 1128.8396  ], [ -97.167404,  202.34435 , 1383.1466  ], [-120.032906,  190.96475 , 1573.3999  ], [-350.77136 ,   43.44217 ,  831.3473  ], [-315.40536 ,  164.55286 , 1049.1747  ], [-230.36957 ,  203.17921 , 1311.9639  ], [  25.895449,  192.35948 , 1296.1571  ], [ 107.10583 ,  116.0503  , 1040.5062  ], [ 129.83815 ,  -48.024902,  850.94806 ], [-112.97075 ,  127.969444, 1477.4457  ]], [[264, 264, 658, 658], [235, 351, 612, 728], [196, 442, 581, 827], [136, 272, 635, 771]])
cgg[0] = agg[1]
cgg[0]
(0, 0, 7, [[-261.92838 ,  186.77643 ,   61.159267], [-186.51082 ,   16.872883,  475.58475 ], [-221.72266 ,  168.16664 ,  890.34485 ], [  41.44908 ,  150.95535 ,  924.1952  ], [ -11.101365,  162.80858 ,  484.58923 ], [ -51.15917 ,  220.1005  ,   35.79458 ], [ -90.1369  ,  159.561   ,  907.27    ], [-133.54684 ,  216.47865 , 1129.5045  ], [ -94.67486 ,  205.02849 , 1383.3682  ], [-119.73675 ,  193.0975  , 1573.7574  ], [-346.56067 ,   55.692474,  832.2764  ], [-317.71588 ,  176.29819 , 1051.3417  ], [-228.90077 ,  205.19804 , 1314.1304  ], [  27.778862,  198.42009 , 1295.205   ], [ 112.37455 ,  139.14867 , 1036.1566  ], [ 142.82643 ,  -12.859319,  837.8236  ], [-111.25674 ,  130.67056 , 1477.5469  ]], [[264, 264, 658, 658], [235, 354, 611, 730], [196, 440, 581, 825], [136, 270, 635, 769]])
g_s_r = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/human36m/extra/human36m-multiview-labels-GTbboxes_subset.npy',allow_pickle=True)
g_s_r = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/human36m/extra/human36m-multiview-labels-GTbboxes_subset.npy',allow_pickle=True).item()
dd = data_2['indexes']
for i in range(len(g_s_r['table'])):
    if i < len(cc):
        g_s_r['table'][i] =  gt_results['table'][cc[i]]
    else:
        g_s_r['table'][i]  = gt_results['table'][dd[i-len(cc)]]
gt_results['table'][cc[0]]
(2, 15, 1972, [[-154.15692 , -166.73433 ,  101.692665], [-188.02461 , -218.61795 ,  559.0082  ], [-214.45801 ,  247.16795 ,  697.1967  ], [  69.89129 ,  232.87415 ,  680.02936 ], [  63.209934, -236.5121  ,  552.0596  ], [  26.372581, -492.96844 ,  170.15668 ], [ -72.2823  ,  240.021   ,  688.613   ], [-135.22006 ,  365.34335 ,  910.1736  ], [ -86.30319 ,  275.35858 , 1149.1604  ], [-104.55392 ,  170.99141 , 1327.898   ], [-440.67908 ,  237.51709 ,  565.99713 ], [-348.21    ,  307.1262  ,  796.4832  ], [-205.83368 ,  292.87894 , 1061.3074  ], [  38.583706,  292.0969  , 1068.9363  ], [ 215.87544 ,  292.0813  ,  825.67993 ], [ 302.08313 ,  181.9871  ,  608.9603  ], [ -90.15825 ,  175.09344 , 1213.8763  ]], [[321, 338, 621, 638], [289, 382, 619, 712], [255, 476, 557, 778], [223, 264, 665, 706]])
cc[0]
75881
rrr = np.array(range(0,1577))
index = []
keypoints_pre_t = []
keypoints_pre_v = []
for i in range(1578):
    indices = np.where(pred_results['indexes'] == i)[0]
    keypoints_pre_t.append(pred_results['keypoints_3d'][indices])
    index.append(indices)
for j in range(129):
    indices = np.where(pred_results['indexes'] == j)[0]
    keypoints_pre_v.append(pred_results_val['keypoints_3d'][indices])
    index.append(indices)
keypoints_pre_t = np.concatenate(keypoints_pre_t ,axis=0)
keypoints_pre_v = np.concatenate(keypoints_pre_v ,axis=0)
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 10, in <module>
IndexError: index 47515 is out of bounds for axis 0 with size 2181
keypoints_pre_t = np.concatenate(keypoints_pre_t ,axis=0)
data['keypoints_3d'] = keypoints_pre_t
data['indexes'] = np.array(range(1578))
with open('train_1.pkl','wb') as f:
    pickle.dump(data,f)
    
range1 = set(range(0, 2193))
range2 = set(range(500, 1114)) 
result = range1 - range2
range2 = set(range(500, 1115)) 
result = range1 - range2
result = np.array(result)
result = range1 - range2
result = np.array(result.tolist())
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 1, in <module>
AttributeError: 'set' object has no attribute 'tolist'
result = np.array(list(result))
range1 = set(range(0, 264))
range2 = set(range(37,172)) 
result_2 = range1 - range2
result_2 = np.array(list(result_2))
data['indexes'] = result
data_2['indexes'] = result_2
index = []
keypoints_pre_t = []
keypoints_pre_v = []
for i in result:
    indices = np.where(pred_results['indexes'] == i)[0]
    keypoints_pre_t.append(pred_results['keypoints_3d'][indices])
    index.append(indices)
    
for i in result_2:
    indices = np.where(pred_results['indexes'] == i)[0]
    keypoints_pre_v.append(pred_results_val['keypoints_3d'][indices])
    index.append(indices)
    
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 3, in <module>
IndexError: index 47515 is out of bounds for axis 0 with size 2181
for i in result_2:
    indices = np.where(pred_results_val['indexes'] == i)[0]
    keypoints_pre_v.append(pred_results_val['keypoints_3d'][indices])
    index.append(indices)
    
keypoints_pre_t = np.concatenate(keypoints_pre_t ,axis=0)
keypoints_pre_v = np.concatenate(keypoints_pre_v ,axis=0)
max(pred_results['indexes'])
159180
min(pred_results['indexes'])
0
data['keypoints_3d'] = keypoints_pre_t
data_2['keypoints_3d'] = keypoints_pre_v
with open('train_1.pkl','wb') as f:
    pickle.dump(data,f)
    
with open('val_1.pkl','wb') as f:
    pickle.dump(data_2,f)
    
g_s_r = np.load('/home/zc/Project/ContextPose-PyTorch-release-master/data/human36m/extra/human36m-multiview-labels-GTbboxes_subset.npy',allow_pickle=True).item()
kp1 = g_s_r['table'][500][3]
kp2 =  keypoints_pre_t[500]
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test = kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
ax1.set_xlim3d(-1500, 1500)
ax1.set_ylim3d(-1500, 1500)
ax1.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
    plotX_p = [keypoints_3d_gt[i][0] for i in group]
    plotY_p = [keypoints_3d_gt[i][1] for i in group]
    plotZ_p = [keypoints_3d_gt[i][2] for i in group]
    ax1.plot(plotX_p, plotY_p, plotZ_p)
plt.show()
kp1 = g_s_r['table'][0][3]
kp2 =  keypoints_pre_t[0]
Exception ignored in: <function Image.__del__ at 0x7ff5213ee5e0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test = kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
ax1.set_xlim3d(-1500, 1500)
ax1.set_ylim3d(-1500, 1500)
ax1.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
    plotX_p = [keypoints_3d_gt[i][0] for i in group]
    plotY_p = [keypoints_3d_gt[i][1] for i in group]
    plotZ_p = [keypoints_3d_gt[i][2] for i in group]
    ax1.plot(plotX_p, plotY_p, plotZ_p)
plt.show()
for i in range(1578):
    kp1 = g_s_r['table'][i][3]
    kp2 =  keypoints_pre_t[i]
    from matplotlib import pylab as plt
    fig = plt.figure()
    keypoints_3d_gt = kp1
    keypoints_test = kp2
    ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax2.set_xlim3d(-1500, 1500)
    ax2.set_ylim3d(-1500, 1500)
    ax2.set_zlim3d(-1500, 1500)
    ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax1.set_xlim3d(-1500, 1500)
    ax1.set_ylim3d(-1500, 1500)
    ax1.set_zlim3d(-1500, 1500)
    for group in CONNECTIVITY_DICT["human36m"]:
        plotX_g = [keypoints_test[i][0] for i in group]
        plotY_g = [keypoints_test[i][1] for i in group]
        plotZ_g = [keypoints_test[i][2] for i in group]
        ax2.plot(plotX_g, plotY_g, plotZ_g)
        plotX_p = [keypoints_3d_gt[i][0] for i in group]
        plotY_p = [keypoints_3d_gt[i][1] for i in group]
        plotZ_p = [keypoints_3d_gt[i][2] for i in group]
        ax1.plot(plotX_p, plotY_p, plotZ_p)
    plt.show()
    
<input>:5: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
^CTraceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 5, in <module>
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/_api/deprecation.py", line 454, in wrapper
    return func(*args, **kwargs)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/pyplot.py", line 840, in figure
    manager = new_figure_manager(
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/pyplot.py", line 384, in new_figure_manager
    return _get_backend_mod().new_figure_manager(*args, **kwargs)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 3574, in new_figure_manager
    return cls.new_figure_manager_given_figure(num, fig)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 3579, in new_figure_manager_given_figure
    return cls.FigureCanvas.new_manager(figure, num)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 1742, in new_manager
    return cls.manager_class.create_with_canvas(cls, figure, num)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backends/_backend_tk.py", line 507, in create_with_canvas
    manager = cls(canvas, num, window)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backends/_backend_tk.py", line 457, in __init__
    super().__init__(canvas, num)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 2832, in __init__
    self.toolbar = self._toolbar2_class(self.canvas)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backends/_backend_tk.py", line 630, in __init__
    self._buttons[text] = button = self._Button(
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backends/_backend_tk.py", line 814, in _Button
    b = tk.Button(
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 2650, in __init__
    Widget.__init__(self, master, 'button', cnf, kw)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 2572, in __init__
    self.tk.call(
KeyboardInterrupt
for i in range(500,1115):
    kp1 = g_s_r['table'][i][3]
    kp2 = keypoints_pre_t[i]
    from matplotlib import pylab as plt
    fig = plt.figure()
    keypoints_3d_gt = kp1
    keypoints_test = kp2
    ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax2.set_xlim3d(-1500, 1500)
    ax2.set_ylim3d(-1500, 1500)
    ax2.set_zlim3d(-1500, 1500)
    ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax1.set_xlim3d(-1500, 1500)
    ax1.set_ylim3d(-1500, 1500)
    ax1.set_zlim3d(-1500, 1500)
    for group in CONNECTIVITY_DICT["human36m"]:
        plotX_g = [keypoints_test[i][0] for i in group]
        plotY_g = [keypoints_test[i][1] for i in group]
        plotZ_g = [keypoints_test[i][2] for i in group]
        ax2.plot(plotX_g, plotY_g, plotZ_g)
        plotX_p = [keypoints_3d_gt[i][0] for i in group]
        plotY_p = [keypoints_3d_gt[i][1] for i in group]
        plotZ_p = [keypoints_3d_gt[i][2] for i in group]
        ax1.plot(plotX_p, plotY_p, plotZ_p)
    # plt.show()
    plt.savefig(f'save/output_{i}.png')
    plt.close(fig)
    
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 27, in <module>
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/pyplot.py", line 1023, in savefig
    res = fig.savefig(*args, **kwargs)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/figure.py", line 3378, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 2366, in print_figure
    result = print_method(
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 2232, in <lambda>
    print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backends/backend_agg.py", line 509, in print_png
    self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/backends/backend_agg.py", line 458, in _print_pil
    mpl.image.imsave(
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/matplotlib/image.py", line 1689, in imsave
    image.save(fname, **pil_kwargs)
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/site-packages/PIL/Image.py", line 2563, in save
    fp = builtins.open(filename, "w+b")
FileNotFoundError: [Errno 2] No such file or directory: '/home/zc/Project/ContextPose-PyTorch-release-master/save/output_500.png'
for i in range(500,1115):
    kp1 = g_s_r['table'][i][3]
    kp2 = keypoints_pre_t[i]
    from matplotlib import pylab as plt
    fig = plt.figure()
    keypoints_3d_gt = kp1
    keypoints_test = kp2
    ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax2.set_xlim3d(-1500, 1500)
    ax2.set_ylim3d(-1500, 1500)
    ax2.set_zlim3d(-1500, 1500)
    ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax1.set_xlim3d(-1500, 1500)
    ax1.set_ylim3d(-1500, 1500)
    ax1.set_zlim3d(-1500, 1500)
    for group in CONNECTIVITY_DICT["human36m"]:
        plotX_g = [keypoints_test[i][0] for i in group]
        plotY_g = [keypoints_test[i][1] for i in group]
        plotZ_g = [keypoints_test[i][2] for i in group]
        ax2.plot(plotX_g, plotY_g, plotZ_g)
        plotX_p = [keypoints_3d_gt[i][0] for i in group]
        plotY_p = [keypoints_3d_gt[i][1] for i in group]
        plotZ_p = [keypoints_3d_gt[i][2] for i in group]
        ax1.plot(plotX_p, plotY_p, plotZ_p)
    # plt.show()
    os.makedirs('output_images', exist_ok=True)
    plt.savefig(f'output_images/output_{i}.png')
    plt.close(fig)
    
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 27, in <module>
NameError: name 'os' is not defined
import os
for i in range(500,1115):
    
    kp1 = g_s_r['table'][i][3]
    kp2 = keypoints_pre_t[i]
    from matplotlib import pylab as plt
    fig = plt.figure()
    keypoints_3d_gt = kp1
    keypoints_test = kp2
    ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax2.set_xlim3d(-1500, 1500)
    ax2.set_ylim3d(-1500, 1500)
    ax2.set_zlim3d(-1500, 1500)
    ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
    ax1.set_xlim3d(-1500, 1500)
    ax1.set_ylim3d(-1500, 1500)
    ax1.set_zlim3d(-1500, 1500)
    for group in CONNECTIVITY_DICT["human36m"]:
        plotX_g = [keypoints_test[i][0] for i in group]
        plotY_g = [keypoints_test[i][1] for i in group]
        plotZ_g = [keypoints_test[i][2] for i in group]
        ax2.plot(plotX_g, plotY_g, plotZ_g)
        plotX_p = [keypoints_3d_gt[i][0] for i in group]
        plotY_p = [keypoints_3d_gt[i][1] for i in group]
        plotZ_p = [keypoints_3d_gt[i][2] for i in group]
        ax1.plot(plotX_p, plotY_p, plotZ_p)
    # plt.show()
    os.makedirs('output_images', exist_ok=True)
    plt.savefig(f'output_images/output_{i}.png')
    plt.close(fig)
    
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Image.__del__ at 0x7ff5213ee5e0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
    
kp1 = gt_results['table'][1115][3]
kp2 = keypoints_pre_t[500]
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test = kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
ax1.set_xlim3d(-1500, 1500)
ax1.set_ylim3d(-1500, 1500)
ax1.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
    plotX_p = [keypoints_3d_gt[i][0] for i in group]
    plotY_p = [keypoints_3d_gt[i][1] for i in group]
    plotZ_p = [keypoints_3d_gt[i][2] for i in group]
    ax1.plot(plotX_p, plotY_p, plotZ_p)
# plt.show()
    
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
for num,i in enumerate(result):
    g_s_r['table'][num][3]=gt_results['table'][i][3]
    
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
ttt =  gt_results['table'][159181:][3]
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
ttt =  gt_results['table'][159181:][3].item()
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
ttt =  gt_results['table'][159181:][3]
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
ttt[0]
5
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
ttt =  gt_results['table'][159181:]
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
for num,i in enumerate(result):
    g_s_r['table'][num+1578][3]=ttt[i][3]
    
Traceback (most recent call last):
  File "/snap/pycharm-community/439/plugins/python-ce/helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 2, in <module>
IndexError: index 1707 is out of bounds for axis 0 with size 1707
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
    
kp1 = g_s_r['table'][1578][3]
kp2 = keypoints_pre_v[0]
from matplotlib import pylab as plt
fig = plt.figure()
keypoints_3d_gt = kp1
keypoints_test = kp2
ax2 = fig.add_subplot(122, projection='3d')  # 122 表示1行2列的网格中的第2个
ax2.set_xlim3d(-1500, 1500)
ax2.set_ylim3d(-1500, 1500)
ax2.set_zlim3d(-1500, 1500)
ax1 = fig.add_subplot(121, projection='3d')  # 122 表示1行2列的网格中的第2个
ax1.set_xlim3d(-1500, 1500)
ax1.set_ylim3d(-1500, 1500)
ax1.set_zlim3d(-1500, 1500)
for group in CONNECTIVITY_DICT["human36m"]:
    plotX_g = [keypoints_test[i][0] for i in group]
    plotY_g = [keypoints_test[i][1] for i in group]
    plotZ_g = [keypoints_test[i][2] for i in group]
    ax2.plot(plotX_g, plotY_g, plotZ_g)
    plotX_p = [keypoints_3d_gt[i][0] for i in group]
    plotY_p = [keypoints_3d_gt[i][1] for i in group]
    plotZ_p = [keypoints_3d_gt[i][2] for i in group]
    ax1.plot(plotX_p, plotY_p, plotZ_p)
plt.show()
    
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7ff5218a0ca0>
Traceback (most recent call last):
  File "/home/zc/anaconda3/envs/myenv/lib/python3.8/tkinter/__init__.py", line 363, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
np.save('human36m-multiview-labels-GTbboxes_subset.npy',g_s_r)

```

