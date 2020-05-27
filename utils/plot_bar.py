import matplotlib.pyplot as plt
import numpy as np

tasks = ['CWRU->IMS', 'CWRU->JNU', 'IMS->CWRU', 'IMS->JNU', 'JNU->CWRU', 'JNU->IMS' ]
x = np.arange(len(tasks))
width = 0.1

# slim=1
# dann    = np.array([80.47, 63.82, 77.85, 67.48, 72.09, 86.67])
# dannent = np.array([85.95, 62.36, 82.26, 65.95, 73.92, 84.19])
# dctln   = np.array([81.98, 66.56, 75.31, 67.40, 74.95, 86.56])
# dannvat = np.array([87.76, 61.89, 72.99, 65.07, 72.31, 91.79])
# dannbnm = np.array([99.90, 77.26, 88.04, 72.61, 91.06, 99.95])
# my      = np.array([99.95, 92.07, 85.85, 81.41, 85.94, 99.97])

# slim=5
dann    = np.array([98.33, 87.55, 99.56, 88.64, 100, 99.79])
dannent = np.array([99.84, 87.40, 99.86, 87.50, 99.98, 99.97])
dctln   = np.array([97.81, 86.20, 100, 89.95, 100, 98.70])
dannvat = np.array([99.74, 86.32, 100, 84.44, 99.98, 99.93])
dannbnm = np.array([99.89, 91.79, 100, 91.84, 100, 100])
my      = np.array([99.91, 96.18, 100, 94.38, 100, 100])


plt.figure(figsize=(10, 10))
# fig, ax = plt.subplots()

# plt.bar(x, dann,  width=width, label='label1',color='darkorange')
# plt.bar(x + width, dannent, width=width, label='label2', color='deepskyblue')
# plt.bar(x + 2*width, dctln, width=width, label='label3', color='deepskyblue', tick_label=tasks)
# plt.bar(x + 3*width, dannvat, width=width, label='label4', color='deepskyblue')
# plt.bar(x + 4 * width, dannbnm, width=width, label='label3', color='green')
# plt.bar(x + 5 * width, my, width=width, label='label4', color='blue')

# c = ['#0780cf','#765005','#701866','#f47a75', '#b6b51f', '#da1f18']
c = ['#99cc00','#fcd300','#008080','#e30039', '#00a8e1', '#0000ff']
plt.bar(x, dann,  width=width, label='label1',color=c[0])
plt.bar(x + 1*width, dannent, width=width, label='label2', color=c[1])
plt.bar(x + 2*width, dctln, width=width, label='label3', color=c[2], tick_label=tasks)
plt.bar(x + 3*width, dannvat, width=width, label='label4', color=c[3])
plt.bar(x + 4*width, dannbnm, width=width, label='label5', color=c[4])
plt.bar(x + 5*width, my, width=width, label='label6', color=c[5])


# plt.title('(a) Performance Comparison with 1 labeled target samples', fontsize=16)
plt.title('(b) Performance Comparison while 5 labeled target samples', fontsize=16)
plt.xticks(size = 14)
plt.yticks(size = 14)

# plt.ylim(ymin=50)
plt.ylim(ymin=70)
plt.xlabel('Tasks', size=14)
plt.ylabel('Accuracy (%)', size=14)
# c = ['red','green','blue','black']
# ds = np.array([[1,0], [2,1], [2,2], [2,3]])
# dt = np.array([[1,0], [2,1], [2,2], [2,3]])
# ds = np.array([[1,0], [2,1], [2,2]])
# dt = np.array([[1,0], [2,1], [2,2]])

# for data, label in ds:
#     plt.scatter(data, label, label=label, marker='o', s=6, c=c[label])


# for data, label in dt:
#     plt.scatter(data, label, label=label, marker='^', s=6, c=c[label])

# plt.legend(['Ball-Source', 'IR-Source', 'Healthy-Source', 'OR-Source', 'Ball-Target', 'IR-Target', 'Healthy-Target', 'OR-Target'], loc='upper center', ncol=4, fontsize='large')
plt.legend(['DANN', 'EntMin', 'DCTLN', 'VADA', 'BNM', 'Proposed'], loc='upper center', ncol=3, fontsize='large')
# plt.savefig('tsne_label.png', dvi=1000)
plt.show()