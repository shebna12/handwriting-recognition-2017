from timeit import timeit
import re

def checker():
	entry = "thoughtful"
	file_text = set(line.strip() for line in open('bigger.txt'))
	for i in file_text:
		if re.match(r'\b' + entry + r'\b', i):
			# return True
			break
	# return False

def checker2():
	entry = "thoughtful"
	for i in open('bigger.txt'):
		if re.match(r'\b' + entry + r'\b', i):
			return True
	return False
if __name__ == '__main__':
    # print(checker())
# t = Timer("checker()", "from __main__ import checker")
# # print(timeit.timeit("checker()", setup="from __main__ import checker"))
# print(t.timeit())


	timeit("checker()",setup="from __main__ import checker",number=10000)