"""
Ball Tree Example
-----------------
"""
import time
import numpy as np
import math
import csv
from os import listdir
from PIL import Image as PImage
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from heapq_max import heappop_max, heappush_max, heappushpop_max, heapify_max

path = "dataset/"
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def distancia(x, y):
    s = 0
    for i in range(len(x)):
        d = x[i] - y[i]
        s += d*d
    return math.sqrt(s)
def mostrar_res(Q):
	for i in Q:
		img = PImage.open(path + i[1])
		img.show()

def mostrar_que(Q):
	img = PImage.open(path + Q)
	img.show()


class BallTree:
    """Simple Ball tree class"""

    # class initialization function
    def __init__(self, data, image):
        self.data = np.asarray(data)
        self.image = np.asarray(image)
        # data should be two-dimensional
        assert self.data.shape[1] == 1023

        self.loc = data.mean(0)
        self.radius = np.sqrt(np.max(np.sum((self.data - self.loc) ** 2, 1)))

        self.child1 = None
        self.child2 = None

        if len(self.data) > 1:
            # sort on the dimension with the largest spread
            largest_dim = np.argmax(self.data.max(0) - self.data.min(0))
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]
            self.image[:] = self.image[i_sort]

            # find split point
            N = self.data.shape[0]
            split_point = 0.5 * (self.data[N / 2, largest_dim]
                                 + self.data[N / 2 - 1, largest_dim])

            # recursively create subnodes
            self.child1 = BallTree(self.data[N / 2:],self.image[N / 2:])
            self.child2 = BallTree(self.data[:N / 2],self.image[:N / 2])

   

    def draw_circle(self, ax, depth=None):

        """Recursively plot a visualization of the Ball tree region"""
        if depth is None or depth == 0:
            circ = Circle(self.loc, self.radius, ec='k', fc='none')
            ax.add_patch(circ)

        if self.child1 is not None:
            if depth is None:
                self.child1.draw_circle(ax)
                self.child2.draw_circle(ax)
            elif depth > 0:
                self.child1.draw_circle(ax, depth - 1)
                self.child2.draw_circle(ax, depth - 1)


    def search(self,t,k,Q,depth=None):
        if distancia(t,self.loc)-self.radius >= distancia(t,Q[0][2]) or depth == 0:
            return Q
        elif self.child1 is None and self.child2 is None:
            c_i=0
            for p in (self.data).tolist():

                if distancia(t,p) < distancia(t,Q[0][2]):
                    point=[distancia(t,p),self.image[c_i],p]
                    heappush_max(Q,point)
                    if len(Q)>k:
                        heappop_max(Q)
                c_i=c_i+1
        else:
            if self.child1 is not None:
                if depth is None:
                    self.child1.search(t,k,Q)
                    self.child2.search(t,k,Q)
                elif depth > 0:
                    self.child1.search(t,k,Q, depth - 1)
                    self.child2.search(t,k,Q, depth - 1)    

#------------------------------------------------------------
# Create a set of structured random points in two dimensions
np.random.seed(0)
canti=805
it=1
ini=3
k=5
tiempo1=[]
cantidad=[]
tiempo2=[]

#for iterator in range(ini,canti,it):
"""
X = np.random.random((5,3)) * 2 - 1
X[:, 1] *= 0.1
X[:, 1] += X[:, 0] ** 2
"""
X=[]
row1=[]
image=[]
counter=0
query=[]
rowq=[]
quer=''

bus=1024*143

with open('index.csv','rb') as f:
	reader = csv.reader(f,delimiter=',',quotechar='"')
	for row in reader:
		
		for i in row:
			if(counter>=bus and counter<bus+1024):
				if counter==bus:
					quer=i
				else:
					rowq.append(float(i))

			if(counter%1024==0):
				if(counter==0):
					image.append(i)
				else:
					X.append(row1)
					row1=[]	
					image.append(i)
				counter=counter+1
			else:
				row1.append(float(i))
				counter=counter+1
	X.append(row1)	
	#pause()

	#for op in range(100):
	query.append(rowq)
print X
X=np.asarray(X)
X1=X
print X1
print image
print len(image)
print len(X1	)

print query
#t1=time.time()
BT = BallTree(X1,image)
#tiempo_1=time.time()-t1
#	t#iempo1.append(tiempo_1)



#otherpoint= np.random.random((1, 1023)) * 2 - 1
otherpoint=np.array(query)
otherpoint=otherpoint.tolist()[0]
print quer
mostrar_que(quer)
Q=[]
for i in range(k):
    point=[distancia(otherpoint,(X.tolist())[i]),image[i],(X.tolist())[i]]
    heappush_max(Q,point)
	

	

	#for op1 in range(100): 
#t2=time.time()
BT.search(otherpoint,k,Q)
mostrar_res(Q)

"""for i in Q:
	print i[1]
	"""	
#print
#print "este es otro"
#print otherpoint


##		tiempo_2=time.time()-t2
	#tiempo2.append(tiempo_2)
	#cantidad.append(iterator) 
	#print ('estoy en %d' %iterator)
"""
plt.ion()
fig=plt.figure()
bx = fig.add_subplot(111)
cx = fig.add_subplot(111)

for level in range(1, 7):
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], s=2)
    BT.draw_circle(ax, depth=level - 1)

    ax.set_title('level %i' % level)
    fig.canvas.draw()
    pause()
    #pause()


fig.canvas.draw()
pause()
print (X)
print (X[:,0])
tarjet, = bx.plot(otherpoint[0],otherpoint[1], 'ro')
print Q
print otherpoint

fig.canvas.draw()
pause()"""
	
"""
P=[]
print Q
for i in Q:
	P.append(i[1])
print P
P=np.array(P)
puntitos, = cx.plot(P[:,0],P[:,1], 'yo')
fig.canvas.draw()
pause()"""
"""formato1=''
formato1=formato1+str(k)+'_build.csv'
with open(formato1, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(cantidad,tiempo1))

formato2=''
formato2=formato2+str(k)+'_query.csv'
with open(formato2, 'w') as f1:
    writer = csv.writer(f1, delimiter='\t')
    writer.writerows(zip(cantidad,tiempo2))
"""

