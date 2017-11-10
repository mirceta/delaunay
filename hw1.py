import math
import numpy as np
import random as rnd

from operator import itemgetter

def pointCloudTestData():
    pts = []
    for i in range(10000):
        x = rnd.random()
        y = rnd.random()
        pts.append([x, y, math.sin(x) + math.cos(y)])
    return pts

class Edges:

    def __init__(self, v1idx, v2idx):
        self.v1 = v1idx
        self.v2 = v2idx
        self.prev = None
        self.next = None

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

        #for i in range(3, len(self.pts)):
        #    self._expand(i)


    # low
    def _expand(self, i):

        head = self.E
        curr = self.E
        while True:
            if self._isPointToRightOfVec(self.pts[i], [self.pts[self.E.v1], self.pts[self.E.v2]]):
                self._connect(i, self.E)
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
        bottomE = Edges(E.v1, pidx) # is this clockwise?
        topE = Edges(pidx, E.v2)

        if E.getPrev().v1 == bottomE.v2 and E.getPrev().v2 == bottomE.v1: # is previous edge the reversed of the bottom one?
            E.insertInFront(topE)
            E = E.getPrev()
            E = E.removeAndReturnNext().removeAndReturnNext()

            # add new tri
        else:
            E.insertInFront(topE)
            E.insertBehind(bottomE)
            E.removeAndReturnNext()

            # add new tri

        # recursive edge legalization


    def _createFirstTri(self):
        p0 = self.pts[0][:2]
        p1 = self.pts[1][:2]
        p2 = self.pts[2][:2]

        if self._isPointLeftOfVecStart(p0, p1, p2):
            self.tris.append([0, 1, 2], [])
            self.E = self._makeEdgeCycle([0, 1, 2])
        else:
            self.tris.append([1, 0, 2], [])
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
        head = Edges(verts[0], verts[1])
        curr = head
        prevVert = verts[1]
        for e in verts[2:]:
            curr.setNext(Edges(prevVert, e))
            prevVert = e
            prev = curr
            curr = curr.getNext()
            curr.setPrev(prev)
        curr.setNext(Edges(verts[-1], verts[0]))
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

