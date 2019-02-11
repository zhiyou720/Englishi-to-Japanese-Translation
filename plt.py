from snsplt import data_generate


def countlen(path):
    with open(path, 'r') as f:
        line = f.readlines()

    length = [len(line[i].split()) for i in range(len(line))]
    distrdict = {length[i]: length.count(length[i]) for i in range(len(length))}  # 统计所有对应长度出现的次数
    return distrdict, length


def ziplen(path1, path2):
    with open(path1, 'r') as f:
        line1 = f.readlines()
    with open(path1, 'r') as f:
        line2 = f.readlines()
    length1 = [len(line1[i].split()) for i in range(len(line1))]
    length2 = [len(line2[i].split()) for i in range(len(line2))]

    return length1, length2


# dict1, length1 = countlen('data/text.en')
# dict2, length2 = countlen('data/text.fr')


def average(seq):
    total = 0
    for item in seq:
        total += item
    return total / len(seq)


def myfind(x, y):
    return [a for a in range(len(y)) if y[a] == x]


def findaverage(length1, length2):
    dict3 = {}
    for item in set(length1):
        findfr = [length2[j] for j in myfind(item, length1)]
        dict3[item] = average(findfr)
    return dict3


import matplotlib.pyplot as plt
import numpy as np

len_tok = lambda sent: len(sent.split(' '))
data1 = data_generate(len_tok, len_tok)
x, y = np.split(data1, [1], axis=1)[0], np.split(data1, [1], axis=1)[1]
x1=list(x.ravel())
y1=list(y.ravel())
distrdict1 = {x1[i]: x1.count(x1[i]) for i in range(len(x1))}  # 统计所有对应长度出现的次数
distrdict2 = {y1[i]: y1.count(y1[i]) for i in range(len(y1))}  # 统计所有对应长度出现的次数





plt.figure(figsize=(16, 10))

ax1 = plt.subplot2grid((10, 12), (0, 0), colspan=5, rowspan=3)
ax1.plot(distrdict1.keys(), distrdict1.values(), 'ro', label="English", markersize=3)
ax1.plot(distrdict2.keys(), distrdict2.values(), 'go', label='Japanese', markersize=3)
ax1.set_xlabel('Sentence lengths (tokens)')
ax1.set_ylabel('Counts')
plt.ylim([0,2000])
plt.xlim([0,70])
# ax1.title('The distribution of sentence lengths in the English and Japanese ')
plt.legend()

ax2 = plt.subplot2grid((10, 12), (4, 0), colspan=5, rowspan=6)
ax2.scatter(x, y, s=1, marker='o', label='Every sentence pair')
k = sum(x * y for x, y in data1) / sum(x ** 2 for x, _ in data1)
print(k)
lines = ax2.plot([0, 60], [0, k * 60], 'r-', lw=1, label='y={0:.2f}*x'.format(k))
ax2.set_xlabel('English sentence lengths (tokens)')
ax2.set_ylabel('Japanese sentence lengths (tokens)')
plt.ylim([0,60])
plt.xlim([0,60])
plt.legend()

len_en_char = lambda sent: len(sent)
len_fr_char = lambda sent: len(sent.replace(' ', ''))
data2 = data_generate(len_en_char, len_fr_char)
x2, y2 = np.split(data2, [1], axis=1)[0], np.split(data2, [1], axis=1)[1]

x3=list(x2.ravel())
y3=list(y2.ravel())
distrdict1 = {x3[i]: x3.count(x3[i]) for i in range(len(x3))}  # 统计所有对应长度出现的次数
distrdict2 = {y3[i]: y3.count(y3[i]) for i in range(len(y3))}  # 统计所有对应长度出现的次数

ax3 = plt.subplot2grid((10, 12), (0, 6), colspan=5, rowspan=3)
ax3.plot(distrdict1.keys(), distrdict1.values(), 'ro', label="English", markersize=1)
ax3.plot(distrdict2.keys(), distrdict2.values(), 'go', label='Japanese', markersize=1)
ax3.set_xlabel('Sentence lengths (characters)')
ax3.set_ylabel('Counts')
plt.ylim([0,1000])
plt.xlim([0,250])
# ax1.title('The distribution of sentence lengths in the English and Japanese ')
plt.legend()


ax4 = plt.subplot2grid((10, 12), (4, 6), colspan=5, rowspan=6)
ax4.scatter(x2, y2, s=1, marker='o', label='Every sentence pair')
plt.ylim([0,250])
plt.xlim([0,250])

k1 = sum(x2 * y2 for x2, y2 in data2) / sum(x2 ** 2 for x2, _ in data2)
print(k1)
lines1 = ax4.plot([0, 250], [0, k1 * 250], 'r-', lw=1, label='y={0:.2f}*x'.format(k1))
ax4.set_xlabel('English sentence lengths (characters)')
ax4.set_ylabel('Japanese sentence lengths (characters)')





plt.legend()
# plt.savefig('1.pdf',format='pdf')



# # plt.figure(2,figsize=(8,4))
# # plt.subplot(412)
# #
# # plt.plot(length1[0:99], label ="English")
# # plt.plot(length2[0:99],label = 'Japanese')
# # # plt.title("/n English and Japanese's correlation")
# # plt.legend()
# # plt.subplot(413)
# # plt.plot(findaverage(length1,length2).keys(), findaverage(length1,length2).values())
# # plt.subplot(414)
# # plt.plot(findaverage(length2,length1).keys(), findaverage(length2,length1).values())
#
plt.show()
