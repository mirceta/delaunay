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


class Delaunay:

    def __init__(self, pts):
        self.pts = pts
        self.tris = [] # 3 points of triangle, 3 pointers to neighbor triangles
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
        while True:
            if self._isPointToRightOfVec(self.pts[i], [self.pts[curr.v1], self.pts[curr.v2]]):
                print('found one!')
                curr = self._connect(i, curr)
            else:
                # don't extend it
                pass
            curr = curr.getNext()
            if curr == head:
                break

    def _connect(self, pidx, E):
        # I could not find a case where the upper edge can have it's reverse in the hull
        # It also doesn't make sense, as it's the only edge that can possibly advance the hull

        # therefore we only check if the bottom one's reverse is in the hull!
        if (len(self.tris) == 3):
            print('')
        bottomE = Edges(E.v1, pidx, len(self.tris)) # is this clockwise?
        topE = Edges(pidx, E.v2, len(self.tris))

        if E.getPrev().v1 == bottomE.v2 and E.getPrev().v2 == bottomE.v1: # is previous edge the reversed of the bottom one?

            bottomE.tri = E.getPrev().tri # then they are the same

            # create new tri
            currTriIdx = len(self.tris)
            self.tris.append(([E.v2, E.v1, pidx], [E.tri, bottomE.tri]))
            self.tris[bottomE.tri][1].append(currTriIdx) # append to bottom neighbor
            self.tris[E.tri][1].append(currTriIdx) # append to top neighbor


            # fix convex hull
            E.insertInFront(topE)
            E = E.getPrev()
            E = E.removeAndReturnNext().removeAndReturnNext()

        else: # 2 new edges

            # create new tri
            currTriIdx = len(self.tris)
            self.tris.append(([E.v2, E.v1, pidx], [E.tri]))
            self.tris[E.tri][1].append(currTriIdx)


            # fix convex hull
            E.insertInFront(topE)
            E.insertBehind(bottomE)
            E = E.removeAndReturnNext()

        # recursive edge legalization

        return E

    def _isEdgeLegal(self, tri1, tri2):
        edges1 = self.tri1[0]
        edges2 = self.tri2[0]

        out_pt1 = list(set(tri2) - set(tri1))
        out_pt2 = list(set(tri1) - set(tri2))
        if self._IsTriDelaunay(tri1, out_pt1[0]) and self._isTriDelaunay(tri2, out_pt2[0]):
            pass
        else:
            self._flipEdges()

    def _flipEdges(self):
        pass


    def _IsTriDelaunay(self, tri1, otherPoint):

        # @Wikipedia: Delaunay Triangulation says that otherPoint is within the circumcircle when this determinant is positive
        A = np.linalg.det([[self.pts[tri1[0]][:2], self.pts[tri1[0]][0] ** 2 + self.pts[tri1[0]][1] ** 2, 1.0],
                          [self.pts[tri1[1]][:2], self.pts[tri1[1]][0] ** 2 + self.pts[tri1[1]][1] ** 2, 1.0],
                          [self.pts[tri1[2]][:2], self.pts[tri1[2]][0] ** 2 + self.pts[tri1[2]][1] ** 2, 1.0],
                          [otherPoint, otherPoint[0] ** 2, otherPoint[1] ** 2, 1.0]])
        if A > 0:
            return False
        else:
            return True

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
