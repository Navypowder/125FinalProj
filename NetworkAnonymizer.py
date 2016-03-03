#!/usr/bin/python

from sets import Set
import random
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




def kAnonymize(graph): # The DP Algorithm

    # cost of setting degree of nodes startIndex to endIndex to d* (median val)
    def anonymizationCost(degrees, startIndex, endIndex):
        dStar = numpy.median(degrees[startIndex : endIndex + 1])
        return sum([abs(dStar - degrees[i]) for i in range(startIndex, endIndex + 1)])

    # Retrieve degrees of graph
    degrees = [ 0 for _ in range(len(graph.vertices))]
    for edge in graph.edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    sorted(degrees) # Sort degrees

    costs = [0] * len(graph.vertices)
    ranges = [None] * len(graph.vertices)

    # first 2k vertices lumped into one degree bin
    for i in xrange(2  * K_VALUE):
        costs[i] = anonymizationCost(degrees, 0, i)
        ranges[i] = (0, i)

    # for rest of vertices, lumped into bins of size k
    for i in xrange(2 * K_VALUE, len(degrees)): # note: if this is too slow, can optimize to be O(kn) instead of O(n^2)
        largeGroupCost = anonymizationCost(degrees, 0, i)
        smallGroupCost = -1
        tValue = -1
        for t in xrange(K_VALUE, i - K_VALUE):
            cost =  costs[t] + anonymizationCost(degrees, t + 1, i)
            if smallGroupCost == -1 or smallGroupCost > cost:
                smallGroupCost = cost
                tValue = t
        if smallGroupCost == -1 or largeGroupCost < smallGroupCost:
            cost[i] = largeGroupCost
            ranges[i] = (0, i)
        else:
            cost[i] = smallGroupCost
            ranges[i] = (tValue + 1, i)

#         costs[i] = min(anonymizationCost(degrees, 0, i), \
#                        min([ costs[t] + anonymizationCost(degrees, t + 1, i) \
#                         for t in xrange(K_VALUE, i - K_VALUE)]))


    # Trace back to get degrees of each node
    newDegrees = [-1] * len(graph.vertices)
    index = len(graph.vertices) - 1
    while(index >= 0):
        startIndex, endIndex = ranges[index]
        dStar = numpy.median(degrees[startIndex : endIndex + 1])
        for i in range(startIndex, endIndex + 1):
            newDegrees[i] = dStar
        index = startIndex - 1

    # Sanity check
    for degree in newDegrees:
        if degree == -1:
            raise Exception("Error: Not all degrees were set to a value")

    return newDegrees

def findBestSwap():
    return

#outputs graph with same degree sequence as initialGraph and high similarity to inputGraph
def greedySwap(initialGraph, inputGraph):
    resultGraph = initialGraph
    c, toRemoveEdge, toAddEdge = findBestSwap(resultGraph)
    while c > 0:
        resultGraph.edges.add(toAddEdge)
        resultGraph.edges.remove(toRemoveEdge)
        c, toRemoveEdge, toAddEdge = findBestSwap(resultGraph)
    return resultGraph

def constructGraph(degrees):
    vertices = range(len(degrees))
    edges = []
    if(sum(degrees) % 2 == 1):
        raise RuntimeError
    while True:
        # Add edges to graph
        for degree in degrees:
            if degree < 0:
                raise RuntimeError
        index = random.randint(0, len(degrees) - 1)
        degrees[index] = 0
        sameDegreeVertices = [ i for i in range(len(degrees)) \
                       if degrees[i] == degrees[index]]
        for vertex in sameDegreeVertices:
            edges.add((index, vertex))
            degrees[vertex] -= 1

        # See if graph is completed
        graphCompleted = not any(degrees)
        if graphCompleted:
            return Graph(vertices, edges) # Return completed graph

def anonymize(graph):

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
file = open("com-youtube.ungraph.txt", "r")

# Go through header
for _ in range(4):
    file.readline()

# Convert file into network object
line = file.readline()
while(line != ""):
    tokens = line.split()
    edge = (int(tokens[0]) - 1, int(tokens[1]) - 1)  # Store as ints
    network.addEdge(edge)
    network.addNode(edge[0])
    network.addNode(edge[1])
    line = file.readline()
file.close()

# Code to anonymize data (I suggest switching to smaller dataset)
anonymizedGraph = anonymize(network.toGraph())
output = open("anonymizedData.txt", "w+")
for edge in anonymizedGraph.edges: # Create newly anonymized dataset
    output.write(str(edge[0]+1) + "\t" + str(edge[1]+1) + "\n")

