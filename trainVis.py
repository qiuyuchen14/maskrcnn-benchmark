import os,sys
# from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

f=open('/home/zoey/nas/zoey/github/maskrcnn-benchmark/checkpoints/test5/log.txt', 'r').readlines()
N=len(f)-1
print(N)
# print(f)

x = []
y1 = []#loss
y2=[] #loss_classifier
y3=[]#loss_box_reg
y4=[]
y5=[]
epoch = []
for i in range(0,N):
	iterationid = f[i].find("iter:")
	classifierid = f[i].find("loss_classifier:")
	boxid = f[i].find("loss_box_reg:")
	segid =  f[i].find("loss_mask")
	poseid = f[i].find("loss_pose")
	lossid=f[i].find("loss:")
	if lossid > -1:

		iteration= int(f[i][iterationid+5:lossid-2])
		if iteration==20 or iteration%100==0:
			loss = float(f[i][lossid+5: lossid+11])
			loss_classifier = float(f[i][classifierid+16: classifierid+22])
			loss_box = float(f[i][boxid+13: boxid+19])
			loss_seg = float(f[i][segid+10: segid+16])
			loss_pose = float(f[i][poseid+10: poseid+16])
			# print("Iter: ", iterid)
			x.append(iteration)
			y1.append(loss)
			y2.append(loss_classifier)
			y3.append(loss_box)
			y4.append(loss_seg)
			y5.append(loss_pose)
			epoch.append(iteration/4000)


plt.plot(x, y1, 'r-', label='total loss')
plt.plot(x, y2, 'g-', label='loss_cls')
plt.plot(x, y3, 'b-', label='loss_bbx')
plt.plot(x, y4, 'y-', label='loss_seg')
plt.plot(x, y5, 'c-', label='loss_pose')
plt.legend(loc='best')
plt.xlabel('Iteration#')
plt.ylabel('Loss')
plt.show()
#     w=f[i].split()
    # l1=w[1:8]
    # l2=w[8:15]
    # try:
    #     list1=[float(x) for x in l1]
    #     list2=[float(x) for x in l2]
    # except ValueError,e:
    #     print "error",e,"on line",i
    # result=stats.ttest_ind(list1,list2)
    # print result[1]
