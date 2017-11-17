import math
import numpy as np
import random as rnd

from operator import itemgetter

def writeObj(points, tris):
    file = open('output.obj', 'w')
    str = ''
    for p in points:
        str += 'v {0:.6f} {1:.6f} {2:.6f}\n'.format(p[0], p[1], p[2])
    str += '\n'
    for t in tris:
        str += 'f {} {} {}\n'.format(t[0][0] + 1, t[0][1] + 1, t[0][2] + 1)
    str = str[:-1]
    file.write(str)
    file.close()
    return True


# test data from http://asrl.utias.utoronto.ca/datasets/3dmap/
def a200_metData():
    file = open('a200_met_000.xyz', 'r').read().split('\n')
    data = []
    for line in file:
        a = line.split(' ')
        if (len(a) < 3):
            continue
        data.append([float(a[0]), float(a[1]), float(a[2])])
    return data


def pointCloudTestData():
    pts = []
    for i in range(10000):
        x = rnd.random() * 50
        y = rnd.random() * 30
        pts.append([x, y, math.sin(x)])
    return pts

def handwrittenTestCase():
    pts = [[0.1, 0.2, 0.0],
           [0.2, 0.4, 0.0],
           [0.3, 0.2, 0.0],
           [0.4, 0.5, 0.0],
           [0.5, 0.1, 0.0],
           [0.565, 0.45, 0.0],
           [0.7, 0.7, 0.0],
           [0.8, 0.3, 0.0]]
    expectedTris = [[1,3,2],
                    [2,3,4],
                    [1,3,5],[4,3,5],
                    [4,5,6],
                    [4,6,7],[2,4,7],
                    [6,5,8],[6,7,8]]
    return pts, expectedTris

class Edges:

    def __init__(self, v1idx, v2idx, tri):
        self.v1 = v1idx
        self.v2 = v2idx
        self.prev = None
        self.next = None
        self.tri = tri

    def setPrev(self, e):
        self.prev = e

    def setNext(self, e):
        self.next = e

    def getPrev(self):
        return self.prev

    def getNext(self):
        return self.next

    def removeAndReturnNext(self):
        ret = self.next
        self.next.setPrev(self.prev)
        self.prev.setNext(self.next)
        self.next = None
        self.prev = None
        return ret

    def insertBehind(self, edge):
        prev = self.prev
        prev.next = edge
        edge.prev = prev
        edge.next = self
        self.prev = edge
        return self

    def insertInFront(self, edge):
        next = self.next
        next.prev = edge
        edge.next = next
        edge.prev = self
        self.next = edge
        return self

    def toString(self):
        return '(' + str(self.v1) + ' ' + str(self.v2) + ')'

    def printCycle(self):
        head = self
        curr = self
        while True:
            print(curr.toString())
            curr = curr.next
            if curr == head:
                break

    def count(self):
        head = self
        curr = self
        cnt = 0
        while True:
            cnt += 1
            curr = curr.next
            if curr == head:
                return cnt


class Delaunay:

    def __init__(self, pts):
        self.pts = pts
        self.tris = [] # 3 pointers to points of triangle, 3 pointers to neighbor triangles
        self.E = None

        self.initialize()

    def initialize(self):

        self.pts = sorted(self.pts, key=itemgetter(0))
        self._addSmallRandomFloatToPts()
        self._createFirstTri()

        for i in range(3, len(self.pts)):
            self._expand(i)


    def _expand(self, i):
        head = self.E
        curr = self.E
        cnt = curr.count()
        for j in range(cnt + 2):
            if self._isPointToRightOfVec(self.pts[i], [self.pts[curr.v1], self.pts[curr.v2]]):
                print('HERE', i)
                curr = self._connect(i, curr)
            else:
                # don't extend it
                pass
            curr = curr.getNext()
            if curr == head:
                break

    def _connect(self, pidx, E):
        bottomE = Edges(E.v1, pidx, len(self.tris)) # is this clockwise?
        topE = Edges(pidx, E.v2, len(self.tris))
        if set([E.getNext().v1, E.getNext().v2]) - set([topE.v1, topE.v2]) == 0 and set([E.getPrev().v1, E.getPrev().v2]) - set([bottomE.v1, bottomE.v2]) == 0:
            print("Circle completed")
            return E

        if set([E.getNext().v1, E.getNext().v2]) - set([topE.v1, topE.v2]) == 0: # not sure?
            print("This actually happened")
            # create new tri
            currTriIdx = len(self.tris)
            self.tris.append([[E.v1, pidx, E.v2], [E.tri, topE.tri]])
            self.tris[E.tri][1].append(currTriIdx)
            self.tris[topE.tri][1].append(currTriIdx)

            # fix convex hull
            E.removeAndReturnNext()
            E.insertInFront(bottomE)
            E.removeAndReturnNext()

        elif E.getPrev().v1 == bottomE.v2 and E.getPrev().v2 == bottomE.v1: # is previous edge the reversed of the bottom one?

            bottomE.tri = E.getPrev().tri # then they are the same

            # create new tri
            currTriIdx = len(self.tris)
            self.tris.append(([E.v2, E.v1, pidx], [E.tri, bottomE.tri]))
            self.tris[bottomE.tri][1].append(currTriIdx) # append to bottom neighbor
            self.tris[E.tri][1].append(currTriIdx) # append to base neighbor

            nbrs = [bottomE.tri, E.tri]

            # fix convex hull
            E.insertInFront(topE)
            E = E.getPrev()
            E = E.removeAndReturnNext().removeAndReturnNext()

            # !!! NEW !!!

            # legalize edge
           # for nbr in nbrs:
            #    if not self._isEdgeLegal(nbr, currTriIdx):
             #       self._flipEdge(nbr, currTriIdx, E)




        else: # 2 new edges

            # create new tri
            currTriIdx = len(self.tris)
            self.tris.append(([E.v2, E.v1, pidx], [E.tri]))
            self.tris[E.tri][1].append(currTriIdx)

            nbr = E.tri


            # fix convex hull
            E.insertInFront(topE)
            E.insertBehind(bottomE)
            E = E.removeAndReturnNext()

            # !!! NEW !!!

            # legalize edge
            #if not self._isEdgeLegal(nbr, currTriIdx):
             #   self._flipEdge(nbr, currTriIdx, E)

        return E


    def _isEdgeLegal(self, tri1, tri2):
        pts1 = self.tris[tri1][0]
        pts2 = self.tris[tri2][0]

        out_pt1 = list(set(pts2) - set(pts1))
        out_pt2 = list(set(pts1) - set(pts2))
        if not (self._IsTriDelaunay(tri1, out_pt1[0]) and self._IsTriDelaunay(tri2, out_pt2[0])):
            return False
        return True

    def _IsTriDelaunay(self, tri_idx, otherPoint_idx):

        tri1 = self.tris[tri_idx][0]

        # @Wikipedia: Delaunay Triangulation says that otherPoint is within the circumcircle when this determinant is positive
        X = [self.pts[tri1[0]][:2] + [self.pts[tri1[0]][0] ** 2 + self.pts[tri1[0]][1] ** 2, 1.0],
            self.pts[tri1[1]][:2] + [self.pts[tri1[1]][0] ** 2 + self.pts[tri1[1]][1] ** 2, 1.0],
            self.pts[tri1[2]][:2] + [self.pts[tri1[2]][0] ** 2 + self.pts[tri1[2]][1] ** 2, 1.0],
            self.pts[otherPoint_idx][:2] + [self.pts[otherPoint_idx][0] ** 2 + self.pts[otherPoint_idx][1] ** 2, 1.0]]
        A = np.linalg.det(X)
        if A > 0:
            return False
        else:
            return True

    def _flipEdge(self, tri1, tri2, hull):


        # 0. store all neighbors
        nbrs = [i for i in self.tris[tri1][1]]
        nbrs += [i for i in self.tris[tri2][1]]


        triset1, triset2 = set(self.tris[tri1][0]), set(self.tris[tri2][0])

        ## 1. find the shared edge
        old_shared = list(triset1 - (triset1 - triset2))

        ## 2. find other points
        new_shared = list(triset1 - set(old_shared)) + list(triset2 - set(old_shared))

        ## 3. construct new tris
        T1new = new_shared + [old_shared[0]]
        T2new = new_shared + [old_shared[1]]

        ## 4. fix orientation (make it CCW)
        T1new = self._fixOrientation(T1new)
        T2new = self._fixOrientation(T2new)
        T1nbrs = []
        T2nbrs = []

        ## 5. fix neighboring relations

        # for T1new
        for i in range(3):
            curr = set([T1new[i], T1new[(i + 1) % 3]])
            if (len(set(T2new) - curr) == 1):
                T1nbrs.append(tri2)
                continue

            for nbr in nbrs:
                if (len(set(self.tris[nbr][0]) - curr)): # they share an edge!
                    # correct in neighbor
                    self._replaceElement(self.tris[nbr][1], tri2, tri1) # if it used to belong to tri2, now it belongs to tri1
                    # add neighbor to self
                    T1nbrs.append(nbr)
                    break

        # for T2new
        for i in range(3):
            curr = set([T2new[i], T2new[(i + 1) % 3]])
            if (len(set(T1new) - curr) == 1):
                T2nbrs.append(tri1)
                continue

            for nbr in nbrs:
                if (len(set(self.tris[nbr][0]) - curr)): # they share an edge!
                    # correct in neighbor
                    self._replaceElement(self.tris[nbr][1], tri1, tri2) # if it used to belong to tri2, now it belongs to tri1
                    # add neighbor to self
                    T2nbrs.append(nbr)
                    break

        ## 6. old tris become new
        self.tris[tri1] = [T1new, T1nbrs]
        self.tris[tri2] = [T2new, T2nbrs]

        ## 7. fix the hull if necessary
        tmp = hull.getPrev()
        for i in range(3):
            edge = set([tmp.v1, tmp.v2])
            if len(set(T1new) - edge) == 1: # this is the current edge's triangle!
                tmp.tri = tri1
            elif len(set(T2new) - edge) == 1:
                tmp.tri = tri2
            tmp = tmp.getNext()




    def _replaceElement(self, list, oldElement, newElement):
        for i, x in enumerate(list):
            if (x == oldElement):
                list[i] = newElement
                return list
        return None

    def _fixOrientation(self, tri):
        if self._isPointToRightOfVec(self.pts[tri[0]], [self.pts[tri[1]], self.pts[tri[2]]]):
            tri.reverse()
            return tri
        else:
            return tri

    def _createFirstTri(self):
        p0 = self.pts[0][:2]
        p1 = self.pts[1][:2]
        p2 = self.pts[2][:2]

        if self._isPointLeftOfVecStart(p0, p1, p2):
            self.tris.append(([0, 1, 2], []))
            self.E = self._makeEdgeCycle([0, 1, 2])
        else:
            self.tris.append(([1, 0, 2], []))
            self.E = self._makeEdgeCycle([1, 0, 2])

    def _isPointToRightOfVec(self, pt, vec):
        if self._isPointLeftOfVecStart(vec[0], vec[1], pt):
            return False
        else:
            return True

    def _isPointLeftOfVecStart(self, p0, p1, p2):
        v = (p2[0] - p0[0]) * (p1[1] - p0[1]) - (p1[0] - p0[0]) * (p2[1] - p0[1])
        if v <= 0:
            return True
        else:
            return False

    def _makeEdgeCycle(self, verts):
        head = Edges(verts[0], verts[1], 0)
        curr = head
        prevVert = verts[1]
        for e in verts[2:]:
            curr.setNext(Edges(prevVert, e, 0))
            prevVert = e
            prev = curr
            curr = curr.getNext()
            curr.setPrev(prev)
        curr.setNext(Edges(verts[-1], verts[0], 0))
        prev = curr
        curr = curr.getNext()
        curr.setPrev(prev)
        curr.setNext(head)
        head.setPrev(curr)
        return head

    def _addSmallRandomFloatToPts(self):
        for i in range(len(self.pts)):
            self.pts[i][0] += rnd.random() * (10 ** -6)
            self.pts[i][1] += rnd.random() * (10 ** -6)

class DelaunayTests:

    def __init__(self, d):
        self.d = d
        self.doTests()

    def doTests(self):
        self.rightHandSideTest()
        self._edgeCycleTest()

    def rightHandSideTest(self):
        p0 = [0.0, 0.0]
        p1 = [1.0, 1.0]
        p2 = [0.25, 0.75]
        p3 = [0.75, 0.25]
        p4 = [0.5, 0.5]
        print(d._isPointToRightOfVec(p2, [p0, p1]) == False)
        print(d._isPointToRightOfVec(p3, [p0, p1]) == True)
        print(d._isPointToRightOfVec(p4, [p0, p1]) == False)

    def _edgeCycleTest(self):
        head = d._makeEdgeCycle([0, 1, 2, 3, 4, 5, 6])
        print(head.toString() == '(0 1)')
        print(head.getNext().toString() == '(1 2)')
        print(head.getPrev().toString() == '(6 0)')
        print(head.getNext().getNext().toString() == '(2 3)')
        print(head.getNext().getPrev().toString() == '(0 1)')


points = pointCloudTestData()
d = Delaunay(points)
DelaunayTests(d)
writeObj(d.pts, d.tris)
