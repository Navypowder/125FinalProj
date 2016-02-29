#!/usr/bin/python

from sets import Set


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


# Construct network object
network = Network() 

# Open file 
file = open("com-youtube.ungraph.txt", "r")

# Go through header
for _ in range(4):
    file.readline()
    
# Convert file into network object 
maxNodeIndex = 1 
line = file.readline()
while(line != ""):
    tokens = line.split()
    nodes = (int(tokens[0]), int(tokens[1]))  # Store as ints 
    network.addEdge(nodes)
    network.addNode(nodes[0])
    network.addNode(nodes[1])
    if max(nodes[0], nodes[1]) > maxNodeIndex:
        maxNodeIndex = max(nodes[0], nodes[1])
    line = file.readline()

# Modify network object 
random.randint() 

