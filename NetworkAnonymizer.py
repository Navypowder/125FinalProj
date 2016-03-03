#!/usr/bin/python

from sets import Set
import random
import math
import numpy

K_VALUE = 5

class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

class Network:
    def __init__(self):
        self.edges = Set()
        self.nodes = Set()
    def addEdge(self, edge):
        self.edges.add(edge)
    def addNode(self, node):
        self.nodes.add(node)
    def removeEdge(self, edge):
        self.edges.remove(edge)
    def isEdge(self, edge):
        return self.edges.contains(edge)
    def isNode(self, node):
        return self.nodes.contains(node)
    def toGraph(self):
        return Graph(list(self.nodes), list(self.edges))

# Takes in a pair of nodes and returns them in the order we standardize
# for edges.
def makeEdge(n1, n2):
    return (min(n1, n2), max(n1, n2))

def kAnonymize(graph): # The DP Algorithm

    print "kAnonymize"

    # cost of setting degree of nodes startIndex to endIndex to d* (median val)
    def anonymizationCost(degrees, startIndex, endIndex):
        dStar = numpy.median(degrees[startIndex : endIndex + 1])
        return sum([abs(dStar - degrees[i]) for i in range(startIndex, endIndex + 1)])

    # Retrieve degrees of graph
    degrees = [ 0 for _ in range(len(graph.vertices))]
    for edge in graph.edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    degrees.sort() # Sort degrees
    print "Retrieved and sorted graph degrees."

    costs = [0] * len(graph.vertices)
    ranges = [None] * len(graph.vertices)

    print "Starting DP implementation"
    # first 2k vertices lumped into one degree bin
    for i in xrange(2  * K_VALUE):
        costs[i] = anonymizationCost(degrees, 0, i)
        ranges[i] = (0, i)

    # for rest of vertices, lumped into bins of size k
    for i in xrange(2 * K_VALUE, len(degrees)): # note: if this is too slow, can optimize to be O(kn) instead of O(n^2)
        largeGroupCost = anonymizationCost(degrees, 0, i)
        smallGroupCost = -1
        tValue = -1
        for t in xrange(K_VALUE, i - K_VALUE + 1):
            cost =  costs[t] + anonymizationCost(degrees, t + 1, i)
            if smallGroupCost == -1 or smallGroupCost > cost:
                smallGroupCost = cost
                tValue = t
        if smallGroupCost == -1 or largeGroupCost < smallGroupCost:
            costs[i] = largeGroupCost
            ranges[i] = (0, i)
        else:
            costs[i] = smallGroupCost
            ranges[i] = (tValue + 1, i)

#         costs[i] = min(anonymizationCost(degrees, 0, i), \
#                        min([ costs[t] + anonymizationCost(degrees, t + 1, i) \
#                         for t in xrange(K_VALUE, i - K_VALUE)]))
    print "Done with DP implementation"


    # Trace back to get degrees of each node
    print "Tracing back to get degrees of each node"
    newDegrees = [-1] * len(graph.vertices)
    index = len(graph.vertices) - 1
    while(index >= 0):
        startIndex, endIndex = ranges[index]
        dStar = numpy.median(degrees[startIndex : endIndex + 1])
        for i in range(startIndex, endIndex + 1):
            newDegrees[i] = dStar
        index = startIndex - 1
    print "Done assigning degrees."

    # Sanity check
    if any([i == -1 for i in newDegrees]):
        raise Exception("Error: Not all degrees were set to a value.")

    return newDegrees

def findBestSwap(inputGraph, anonymizedGraph):

    print "findBestSwap"

    # For computational purposes we only examien some subset of the edges
    # in the graph.
    examineThreshold = math.log(len(graph.edges), 2)
    numExamined = 0

    c = 0
    toAddEdges = ((-1, -1), (-1, -1))
    toRemoveEdges = ((-1, -1), (-1, -1))

    while numExamined < examineThreshold and c != 2:
        [e1, e2] = random.sample(graph.edges, 2)

        # if any of the nodes are the same, we won't consider this edge pair
        if any([i in e2 for i in e1]):
            continue

        swapSets = [ [ e1, e2, makeEdge(e1[0], e2[0]), makeEdge(e1[1], e2[1]) ],
                     [ e1, e2, makeEdge(e1[0], e2[1]), makeEdge(e1[1], e2[0]) ] ]
        for swap in swapSets:
            for edge in swap[2:]:
                if anonymizedGraph.edges.contains(edge):
                    continue

            def swapDifference(swaps, inputGraph):
                _c = 0
                if inputGraph.edges.contains(swaps[0]):
                    _c -= 1
                if inputGraph.edges.contains(swaps[1]):
                    _c -= 1
                if inputGraph.edges.contains(swaps[2]):
                    _c += 1
                if inputGraph.edges.contains(swaps[3]):
                    _c += 1
                return _c

            if swapDifference(swap, inputGraph) > c:
                c = swapDifference(swap, inputGraph)
                toAddEdges = tuple(swaps[2:])
                toRemoveEdges = tuple(swaps[:2])

        numExamined += 1

    return c, toRemoveEdges, toAddEdges

#outputs graph with same degree sequence as initialGraph and high similarity to inputGraph
def greedySwap(initialGraph, inputGraph):

    print "greedySwap"

    resultGraph = initialGraph
    c, toRemoveEdge, toAddEdge = findBestSwap(inputGraph, initialGraph)

    while c > 0:
        for edge in toAddEdge:
            resultGraph.edges.add(edge)
        for edge in toRemoveEdge:
            resultGraph.edges.remove(edge)
        c, toRemoveEdge, toAddEdge = findBestSwap(resultGraph, inputGraph)

    return resultGraph

def constructGraph(degrees):

    print "constructGraph"

    edges = []
    if(sum(degrees) % 2 == 1):
        raise RuntimeError
    while True:
        # Add edges to graph
        for degree in degrees:
            if degree < 0:
                raise RuntimeError

        index = random.randint(0, len(degrees) - 1)
        d = degrees[index]
        degrees[index] = 0
        degDLargest = numpy.argsort(degrees)[:d]
        for vertex in degDLargest:
            edges.add(min(index, vertex), max(index, vertex))
            degrees[vertex] -= 1

        # See if graph is completed
        graphCompleted = not any(degrees)
        if graphCompleted:
            return Graph(range(len(degrees)), edges) # Return completed graph

def anonymize(graph):

    print "anonymize"

    # get k-anonymous degrees for each node
    degreeSequence = kAnonymize(graph)
    try:
        # construct graph with desired degree sequence
        kAnonymousGraph = constructGraph(degreeSequence)
        return greedySwap(kAnonymousGraph, graph)
    except RuntimeError:
        print "Error constructing graph"

# Construct network object
network = Network()


# Open file
file = open("p2p-Gnutella08.txt", "r")

# Go through header
for _ in range(4):
    file.readline()

# Convert file into network object
line = file.readline()
while(line != ""):
    tokens = line.split()
    #edge = (int(tokens[0]) - 1, int(tokens[1]) - 1)  # Store as ints
    edge = (int(tokens[0]), int(tokens[1]))  # Store as ints
    network.addEdge(edge)
    network.addNode(edge[0])
    network.addNode(edge[1])
    line = file.readline()
file.close()
print "Done reading file."

# Code to anonymize data (I suggest switching to smaller dataset)
anonymizedGraph = anonymize(network.toGraph())
output = open("anonymizedData.txt", "w+")
for edge in anonymizedGraph.edges: # Create newly anonymized dataset
    output.write(str(edge[0]+1) + "\t" + str(edge[1]+1) + "\n")

