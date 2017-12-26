"""
Ball Tree Example
-----------------
"""
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from heapq_max import heappop_max, heappush_max, heappushpop_max, heapify_max


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def distancia(x, y):
    s = 0
    for i in range(len(x)):
        d = x[i] - y[i]
        s += d*d
    return math.sqrt(s)


class BallTree:
    """Simple Ball tree class"""

    # class initialization function
    def __init__(self, data):
        self.data = np.asarray(data)

        # data should be two-dimensional
        assert self.data.shape[1] == 2

        self.loc = data.mean(0)
        self.radius = np.sqrt(np.max(np.sum((self.data - self.loc) ** 2, 1)))

        self.child1 = None
        self.child2 = None

        if len(self.data) > 1:
            # sort on the dimension with the largest spread
            largest_dim = np.argmax(self.data.max(0) - self.data.min(0))
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]

            # find split point
            N = self.data.shape[0]
            split_point = 0.5 * (self.data[N / 2, largest_dim]
                                 + self.data[N / 2 - 1, largest_dim])

            # recursively create subnodes
            self.child1 = BallTree(self.data[N / 2:])
            self.child2 = BallTree(self.data[:N / 2])

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
        if distancia(t,self.loc)-self.radius >= distancia(t,Q[0][1]) or depth is None or depth == 0:
            return Q
        elif self.child1 is None and self.child2 is None:
            for p in (self.data).tolist():

                if distancia(t,p) < distancia(t,Q[0][1]):
                    point=[distancia(t,p),p]
                    heappush_max(Q,point)
                    if len(Q)>k:
                        heappop_max(Q)
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
X = np.random.random((10, 2)) * 2 - 1
X[:, 1] *= 0.1
X[:, 1] += X[:, 0] ** 2
k=3
Q=X[1:k+1]
Q=Q.tolist()
heapify_max(Q)

BT = BallTree(X)


otherpoint= np.random.random((1, 2)) * 2 - 1
otherpoint=otherpoint.tolist()[0]
Q=[]
for i in range(k):
    point=[distancia(otherpoint,(X.tolist())[i]),(X.tolist())[i]]
    heappush_max(Q,point)

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
pause()
BT.search(otherpoint,k,Q,15)
P=[]
print Q
for i in Q:
    P.append(i[1])
print P
P=np.array(P)
puntitos, = cx.plot(P[:,0],P[:,1], 'yo')
fig.canvas.draw()
pause()