from flask import Flask
from time import sleep
from threading import Thread

app = Flask(__name__)

# After 5 seconds, 'This is a test is printed in the terminal'
def slow_function(some_object):
    sleep(5)
    print(some_object)

def async_slow_function(some_object):
	# args always expects a list
    thr = Thread(target=slow_function, args=[some_object])
    thr.start()
    return thr

@app.route('/')
def index():
    some_object = 'This is a test'
    async_slow_function(some_object)
    return 'hello'  # hello can be shown on the screen

if __name__ == '__main__':
    app.run()
    # Run this example in your browser, 
    # and you'll notice that the index 
    # route now loads instantly. 
    # But wait another five seconds and sure enough, 
    # the test message will print in your terminal.