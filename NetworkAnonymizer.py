#!/usr/bin/python
# forked and refined from todd9527

# from sets import Set
import random
import math
import numpy

K_VALUE = 4

class Graph:
    def __init__(self, vertices, edges):  # 定义图类，包括节点，边
        self.vertices = vertices
        self.edges = edges

class Network:
    def __init__(self):  # 创建边集，点集
        self.edges = set()
        self.nodes = set()
    def addEdge(self, edge):  # 增加边
        self.edges.add(edge)
    def addNode(self, node):  # 增加点
        self.nodes.add(node)
    def removeEdge(self, edge):  # 移除边
        self.edges.remove(edge)
    def isEdge(self, edge):  # 判断图是否包含边
        return self.edges.contains(edge)
    def isNode(self, node):  # 判断图是否包含点
        return self.nodes.contains(node)
    def toGraph(self):  # 返回图：包括节点列表和边列表
        return Graph(list(self.nodes), list(self.edges))

# Takes in a pair of nodes and returns them in the order we standardize
# for edges.
def makeEdge(n1, n2):  # 构造边，数小在前，数大在后
    return (min(n1, n2), max(n1, n2))

def kAnonymize(graph): # The DP Algorithm  利用动态规划算法实现图匿名并返回度序列

    print("kAnonymize")

    # cost of setting degree of nodes startIndex to endIndex to d* (median val)
    def anonymizationCost(degrees, startIndex, endIndex):
        dStar = numpy.median(degrees[startIndex : endIndex + 1])
        return sum([abs(dStar - degrees[i]) for i in range(startIndex, endIndex + 1)])

    # Retrieve degrees of graph
    degrees = [ 0 for _ in range(len(graph.vertices))]   # degree列表所有元素都设为0，长度为图节点个数
    for edge in graph.edges:  # 图中各边：两端节点度数各+1
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    degrees.sort() # Sort degrees
    print ("Retrieved and sorted graph degrees.")

    costs = [0] * len(graph.vertices)
    ranges = [None] * len(graph.vertices)

    print ("Starting DP implementation")
    # first 2k vertices lumped into one degree bin
    # for i in xrange(2  * K_VALUE):
    for i in range(2 * K_VALUE):
        costs[i] = anonymizationCost(degrees, 0, i)
        ranges[i] = (0, i)

    print ("total number of its to do = ", (len(degrees) - 2 * K_VALUE))
    # for rest of vertices, lumped into bins of size k
    # for i in xrange(2 * K_VALUE, len(degrees)): # note: if this is too slow, can optimize to be O(kn) instead of O(n^2)
    for i in range(2 * K_VALUE, len(degrees)): # note: if this is too slow, can optimize to be O(kn) instead of O(n^2)
        #largeGroupCost = anonymizationCost(degrees, 0, i)
        minCost = -1 
        #smallGroupCost = -1
        tValue = -1
        # smallSet = [t for t in xrange(max(K_VALUE, i - 2 * K_VALUE + 1), i - K_VALUE)]
        smallSet = [t for t in range(max(K_VALUE, i - 2 * K_VALUE + 1), i - K_VALUE)]
        for t in smallSet:
            cost = costs[t] + anonymizationCost(degrees, t + 1, i)
            if minCost == -1 or cost < minCost: 
                tValue = t
                minCost = cost
        costs[i] = minCost 
        ranges[i] = (tValue + 1, i)
#         for t in xrange(K_VALUE, i - K_VALUE + 1):
#             cost =  costs[t] + anonymizationCost(degrees, t + 1, i)
#             if smallGroupCost == -1 or smallGroupCost > cost:
#                 smallGroupCost = cost
#                 tValue = t
#         if smallGroupCost == -1 or largeGroupCost < smallGroupCost:
#             costs[i] = largeGroupCost
#             ranges[i] = (0, i)
#         else:
#             costs[i] = smallGroupCost
#             ranges[i] = (tValue + 1, i)

#         costs[i] = min(anonymizationCost(degrees, 0, i), \
#                        min([ costs[t] + anonymizationCost(degrees, t + 1, i) \
#                         for t in xrange(K_VALUE, i - K_VALUE)]))
    print ("Done with DP implementation")


    # Trace back to get degrees of each node
    print ("Tracing back to get degrees of each node")
    newDegrees = [int(-1)] * len(graph.vertices)
    index = len(graph.vertices) - 1
    while(index >= 0):
        startIndex, endIndex = ranges[index]
        dStar = numpy.median(degrees[startIndex : endIndex + 1])
        for i in range(startIndex, endIndex + 1):
            newDegrees[i] = dStar
        index = startIndex - 1
    print ("Done assigning degrees.")

    # Sanity check
    if any([i == -1 for i in newDegrees]):
        raise Exception("Error: Not all degrees were set to a value.")
    
   
    return sorted(newDegrees , reverse=True)  # 正序排列匿名度序列

def findBestSwap(inputGraph, anonymizedGraph):  # 输入分别为原始图与匿名图

    print("findBestSwap")

    # For computational purposes we only examien some subset of the edges
    # 算力原因进检验边的部分子集
    # in the graph.
    examineThreshold = math.log(len(anonymizedGraph.edges), 2)  # 计算以2为底，边个数的对数log2(len(anonymizedGraph.edges)
    numExamined = 0

    c = 0
    toAddEdges = ((-1, -1), (-1, -1))
    toRemoveEdges = ((-1, -1), (-1, -1))

    while numExamined < examineThreshold and c != 2:  # 检测数量 < 检测门限
        [e1, e2] = random.sample(anonymizedGraph.edges, 2)  # 从边集中随机挑出两个组成边对[e1, e2]

        # if any of the nodes are the same, we won't consider this edge pair
        # 如果两条边中有重复节点，将不考虑这对边
        if any([i in e2 for i in e1]):  # e1，e2两条边中节点有重复
            print("HELLO MF")
            continue

    # 边交换，两条边本来是e1[0]e1[1]，e2[0]e2[1]，交换得到四条边e1[0]e2[0],e1[0]e2[1],e1[1]e2[0],e1[1]e2[1]，但只有0011和0110会真正交换
        swapSets = [[e1, e2, makeEdge(e1[0], e2[0]), makeEdge(e1[1], e2[1])],
                     [e1, e2, makeEdge(e1[0], e2[1]), makeEdge(e1[1], e2[0])]]
        for swap in swapSets:
            for edge in swap[2:]:  # 从每一个元组的第三个元素到最后
                if edge in anonymizedGraph.edges:  # 匿名边集中已包含该边
                    print ("ME AGAIN MFERS")
                    continue

            def swapDifference(swaps, inputGraph):
                _c = 0  # 设_c = 0
                if swaps[0] in inputGraph.edges:  # swaps组合中第一条边：原边1在原图的边中，_c-1，去掉了原图中有的边，减分
                    _c -= 1
                if swaps[1] in inputGraph.edges:  # swaps组合中第二条边：原边2在原图的边中，_c-1，去掉了原图中有的边，减分
                    _c -= 1
                if swaps[2] in inputGraph.edges:  # swaps组合中第三条边：新边1在原图的边中，_c+1，增加了原图中有的边，加分
                    _c += 1
                if swaps[3] in inputGraph.edges:  # swaps组合中第四条边：新边2在原图的边中，_c+1，增加了原图中有的边，加分
                    _c += 1
                return _c

            if swapDifference(swap, inputGraph) > c:  # 换边后有加分，即换边后比换边前更接近原图中的边集
                c = swapDifference(swap, inputGraph)
                toAddEdges = tuple(swap[2:])  # 将元组(交换边1，交换边2)赋给toAddEdges
                toRemoveEdges = tuple(swap[:2])  # 将元组(去除边1，去除边2)赋给toRemoveEdges

        print("numExamined %d" % numExamined)  # 检测次数+1，迟早达到算力规定的界限examineThreshold = log2(边个数)
        numExamined += 1

    print("Returning a swap improvement of %d" % c)
    print("Edges:")
    print(toRemoveEdges)
    print(toAddEdges)

    return c, toRemoveEdges, toAddEdges  # 返回c值，去除边元组，增加边元组

#outputs graph with same degree sequence as initialGraph and high similarity to inputGraph
def greedySwap(initialGraph, inputGraph):  # 输入分别为initialGraph初始匿名图, inputGraph原始图

    print("greedySwap")

    resultGraph = initialGraph  # 将最开始的匿名图设为结果图
    c, toRemoveEdge, toAddEdge = findBestSwap(inputGraph, initialGraph)  # 寻找最佳交换：输入为(原始图，初始匿名图)

    while c > 0:  # 当c > 0，匿名图的边更接近原图时
        for edge in toAddEdge:
            resultGraph.edges.add(edge)
        for edge in toRemoveEdge:
            resultGraph.edges.remove(edge)
        c, toRemoveEdge, toAddEdge = findBestSwap(resultGraph, inputGraph)

    return resultGraph

def constructGraph(degrees):

    print ("constructGraph")

    edges = []
    if(sum(degrees) % 2 == 1):
        raise TypeError

    i = 1
    unseenIndices = set(range(len(degrees)))
    while True:
        print ("Adding node %d" % i)

        # Add edges to graph
        for degree in degrees:
            if degree < 0:
                raise RuntimeError

        index = random.choice(tuple(unseenIndices))

        d = degrees[index]
        degrees[index] = 0
        unseenIndices.remove(index)

        degDLargest = numpy.argsort(degrees)[::-1][:d]
        for vertex in degDLargest:
            edges.append(makeEdge(index, vertex))
            degrees[vertex] -= 1

        # See if graph is completed
        i += 1
        graphCompleted = not any(degrees)
        if graphCompleted:
            return Graph(range(len(degrees)), edges) # Return completed graph

def anonymize(graph):

    print("anonymize")

    # get k-anonymous degrees for each node
    degreeSequence = kAnonymize(graph)  # 得到新的匿名度序列(通过贪婪算法或动态规划)
    probeIndex = 0
    while True:
        try:
            # construct graph with desired degree sequence
            kAnonymousGraph = constructGraph(degreeSequence)
            return greedySwap(kAnonymousGraph, graph)
        except TypeError:
            print("A: When you fail, get back up and try again.")
            degreeSequence[probeIndex] += 1
            probeIndex += 1
        except RuntimeError:
            print("B: When you fail, get back up and try again.")
            degreeSequence[probeIndex] += 2
            probeIndex += 1

# Construct network object
network = Network()


# Open file
file = open("karate2.txt", "r")

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
print ("Done reading file.")

# Code to anonymize data (I suggest switching to smaller dataset)
anonymizedGraph = anonymize(network.toGraph())
output = open("anonymizedData.txt", "w+")
print ("Opened the file")
for edge in anonymizedGraph.edges:  # Create newly anonymized dataset
    output.write(str(edge[0]+1) + "\t" + str(edge[1]+1) + "\n")
print ("Done writing the FILE MTHERFUCKERSSSSZZZZZZZ")
