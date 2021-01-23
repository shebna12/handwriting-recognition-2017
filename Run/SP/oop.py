class pet:
	number_of_legs = 0
	def sleep(self):
		print("zzzz")
	def count_legs(self):
		print("I have a %d legs." %self.number_of_legs)

class dog(pet):
	def sleep(self):
		print("ngawww")
	def bark(self):
		print("woof")


doug = dog()
doug.number_of_legs = 4
doug.count_legs()
doug.sleep()
print(300//1000)