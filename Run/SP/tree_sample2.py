import ete3
from ete3 import Tree, TreeStyle
import pickle
# from time import Time

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

name2node = load_obj("name2node")
tree = Tree("dictree.nw",format=8)
# print(tree.get_ascii())

# name2node = {}
ts = TreeStyle()
though_node = name2node["thoughtful"]
though2_node = name2node["ark"]
print (though_node.get_ascii())
# tree.show(ltree_style=ts)

# tree.show()
# import sys
# sys.exit(0)
# # You can rebuild words under a given node
def reconstruct_fullname(node):
    name = []
    while node.up:
        name.append(node.name)
        print("Name: ",node.name)

        node = node.up

    name = ''.join(reversed(name))
    return name

# from multiprocessing.dummy import Pool as ThreadPool 
# pool = ThreadPool(4) 

for leaf in though_node.iter_leaves():
    print("LEAF NAME: ",leaf.name)
    print (reconstruct_fullname(leaf))
print("----------------------------------------------")
for leaf in though2_node.iter_leaves():
    print("LEAF NAME: ",leaf.name)
    print (reconstruct_fullname(leaf))

