import re
hop = "From stephen.marquard@uct.ac.za Sat Jan  5 09:14:16 2008"
# y = re.findall('(\S+)@+',hop)
# print(y)


x = "<span>Please click this ad.</span>"
y = re.findall('span(.+)span', x) 
print(y)