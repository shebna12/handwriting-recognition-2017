import ete3
from ete3 import Tree
import pickle
from spellchecker import correction
def backtrack(current_node,fullname):
    print("Backtracking from: ",current_node.name, fullname)
    current_node = get_parent(current_node)
    fullname = fullname[:-1]
    print("After Backtracking: ",current_node.name, fullname)
    return current_node,fullname

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Lets keep tree structure indexed
name2node = {}
# Make sure there are no duplicates
words = set(line.strip() for line in open('short_dict.txt'))
print(len(words))

# words = set(words)
# Populate tree
tree = Tree()
for wd in words:

    # If no similar words exist, add it to the base of tree
    target = tree

    # Find relatives in the tree
    for pos in range(len(wd), -1, -1):
        # print("pos:",pos)
        root = wd[:pos]
        # print("root: ",root)
        # An existing substring is in the index dict. So you start adding children on that part
        if root in name2node:
            # print("if root in name2node: ",name2node[root])
            target = name2node[root]
            break

    # Add new nodes as necessary
    fullname = root 
    # print("init fullname: ",fullname)
    # Start on the position kung saan may existing substring OR from the starting letter of the word talaga
    for letter in wd[pos:]:
        # print("Letter: ",letter)
        fullname += letter 
        # print("fullname: ",fullname)
        new_node = target.add_child(name=letter, dist=1.0)
        name2node[fullname] = new_node
        # print("name2node: ",name2node[fullname])
        target = new_node
        # print("target: ",target.name)

# tree.render("mytree.png",h=800,w=600,units="px")
save_obj(name2node,"short_dict")
# Print structure
# print (name2node.get_ascii())
# tree.write(format=8, outfile="dictree.nw")
import sys
sys.exit(0)
# You can also use all the visualization machinery from ETE
# (http://packages.python.org/ete2/tutorial/tutorial_drawing.html)
# tree.show()

# You can find, isolate and operate with a specific node using the index

# t.show()
wh_node = name2node["though"]

# print (wh_node.get_ascii())

# You can rebuild words under a given node
def recontruct_fullname(node):
    name = []
    while node.up:
        name.append(node.name)
        node = node.up
    name = ''.join(reversed(name))
    return name

# for leaf in wh_node.iter_leaves():
    # print (recontruct_fullname(leaf))

def get_parent(node):
    parent = node.up
    return parent
#####Pseudocode-ish for checking the existence of a substring 
print("**************START OF CODE****************")
# potential_word = ['d','u','m','b']


pot_tree_style = TreeStyle()
pot_tree_style.show_leaf_name = False

words = ['djmb', 'dumb', 'domb', 'ojmb', 'oumb', 'oomb', 'bjmb', 'bumb', 'bomb']
# words = ['kisseo', 'kisseb', 'kissed']
# words = ['udgment', 'udgaent', 'udghent']
potential_word_tree = Tree()
pot2node = {}

def my_layout(node):
        F = TextFace(node.name, tight_text=True)
        add_face_to_node(F, node, column=0, position="branch-right")
pot_tree_style.layout_fn = my_layout
for wd in words:

    # If no similar words exist, add it to the base of tree
    target = potential_word_tree

    # Find relatives in the tree
    for pos in range(len(wd), -1, -1):
        # print("pos:",pos)
        root = wd[:pos]
        # print("root: ",root)
        # An existing substring is in the index dict. So you start adding children on that part
        if root in pot2node:
            # print("if root in name2node: ",name2node[root])
            target = pot2node[root]
            break

    # Add new nodes as necessary
    fullname = root 
    # print("init fullname: ",fullname)
    # Start on the position kung saan may existing substring OR from the starting letter of the word talaga
    for letter in wd[pos:]:
        # print("Letter: ",letter)
        fullname += letter 
        # print("fullname: ",fullname)
        new_node = target.add_child(name=letter, dist=1.0)
        pot2node[fullname] = new_node
        # print("name2node: ",name2node[fullname])
        target = new_node


node_fullname = potential_word_tree.get_tree_root().name
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# potential_word_tree.render("potree.png",w=600,units="px",tree_style=pot_tree_style)
root = potential_word_tree.get_tree_root()
print(root.name)
print("root kids: ",root.get_children())
# startnode = root
# for child in startnode.get_children():
#     print("parent: ",startnode.name)
#     print("child: ", child.name)
#     startnode = child

def explore_tree(current_node,visited,fullname):
    # exists= True    
    for child in current_node.get_children():
        print("Got this child: ", child.name)

        if(child not in visited):
            print("gonna visit this kid!")
            # exit_signal = False
            visited.append(child)
            if(fullname == "$"):
                print("virgin")
                potential_fullname = child.name
            else:
                potential_fullname = fullname + child.name
                print("Checking potential: ", potential_fullname)
            
            try:
                exists = name2node[potential_fullname]
            except KeyError as okError:
                exists = False
            if(not exists):
                print("Shit doesn't exist in the dict.")
                potential_fullname = fullname
                print("Children of current node are: ",current_node.get_children())
                if(set(current_node.get_children()).issubset(visited)):
                    print("About to backtrack bruh!")
                    current_node,fullname = backtrack(current_node,fullname)
            else:
                print("Great! Proceed.")
                current_node = child
                fullname = potential_fullname
                break
           

    return current_node,visited,fullname,exists
init_node = root
current_node = init_node
visited = []
fullname = "$" 
exit_signal = False
print("Length of words: ",len(words[0]))
while((len(fullname)) != len(words[0]) and exit_signal==False):
    print("Exit signal: ",exit_signal)
    print("fullname: ",fullname)
    try:
        current_node,visited,fullname,exists= explore_tree(current_node,visited,fullname) 
        print("Exit signal = ", exit_signal)
    except UnboundLocalError as okError2:
        print("Got an UNBOUNDLOCALERROR")
        fullname = correction(words[0])
        exit_signal = True
        continue

    print("Current Node: ",current_node.name)
    print("Visited nodes: ",visited)
    print("Fullname: ",fullname)
    # if(current_node.get_children() in visited or ):
    #     current_node,fullname = backtrack(current_node,fullname)
# print(potential_word_tree.get_ascii)
print("FOUND ANSWER: ", fullname)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")









