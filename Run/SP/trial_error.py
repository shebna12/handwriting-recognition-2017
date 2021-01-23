x = ['KISSED YOU THOUGHT WAS DUMB', 'BUT THINK THAT SOMEBODY POOR', 'JUDGMENT SEEING NAME IN BLACK', 'AND WHITE LIKE MAKING LOVE', 'WITH YOU ALL NIGHT FEEL MUCH', 'BETTER THAN BEFORE MAYBE SHE', 'WHAT YOU PREFER BUT LAST YEAR', 'WAS HER MAYBE WOULD CHANGE']
print(x)
# import itertools as it
# shrinked_labels = list(it.chain(*x))
# nlab = ''.join(shrinked_labels)
# # print(*shrinked_labels)
# # shrinked_labels = list(*shrinked_labels)
# # separated_labels = shrinked_labels.split()
# print(shrinked_labels)
# print(nlab)
# acc = 99
new_words = []
for line in x:
	line = line.split()
	# new_words = []
	print(line)
	for word in line:
		print("WORD: ",word)
		new_words.append(word)



print(new_words)