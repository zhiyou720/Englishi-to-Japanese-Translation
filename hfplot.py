from matplotlib import pyplot as plt

# deliveroo=[16,20,28,19,25,17,23,32,8,24]
deliveroo = [4,5,-3,6,0,8,2,-7,10,1]
# justeat=[35,47,52,38,54,29,62,42,48,60]
justeat = [0,-2,-7,7,-9,16,-17,3,10,0]
x = [0,1,2,3,4,5,6,7,8,9]
xx = [0,1,2,3,4,5,6,7,8,9]
y = [0,0,0,0,0,0,0,0,0,0]
date=['8th','9th','10th','11th','12th','13th','14th ','15th','16th','17th']
fig, ax = plt.subplots()
# plt.xlim(0,10)
plt.xticks(x,date,rotation=0)
# plt.ylim(0,70)
plt.ylim(-30,30)
# plt.plot(12,3.58,'o',label='baseline given')
plt.plot(x,deliveroo,label='Deliveroo')
plt.plot(justeat,label='Just Eat')
# plt.plot(xx,y,color='black')
plt.xlabel("Experiments' date in April")
plt.ylabel('minutes')
plt.legend()
plt.grid()
plt.show()
