import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
plt.title('title', fontsize=16)
plt.xticks(size = 14)
plt.yticks(size = 14)

c = ['red','green','blue','black']
# ds = np.array([[1,0], [2,1], [2,2], [2,3]])
# dt = np.array([[1,0], [2,1], [2,2], [2,3]])
ds = np.array([[1,0], [2,1], [2,2]])
dt = np.array([[1,0], [2,1], [2,2]])

for data, label in ds:
    plt.scatter(data, label, label=label, marker='o', s=6, c=c[label])


for data, label in dt:
    plt.scatter(data, label, label=label, marker='^', s=6, c=c[label])

# plt.legend(['Ball-Source', 'IR-Source', 'Healthy-Source', 'OR-Source', 'Ball-Target', 'IR-Target', 'Healthy-Target', 'OR-Target'], loc='upper center', ncol=4, fontsize='large')
plt.legend(['Healthy-Source', 'OR-Source', 'IR-Source', 'Healthy-Target', 'OR-Target', 'IR-Target'], loc='upper center', ncol=3, fontsize='large')
plt.savefig('legend/tsne_label.png', dvi=1000)