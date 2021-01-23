import socket

s = socket.socket()
host = socket.gethostname()
port = 10101
s.bind((host,port))

s.listen(10)
while True:
	c, addr = s.accept()
	print("Succesfully made a connection from",addr)
	c.send("Thank you for connecting")
	c.close()