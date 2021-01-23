#!/usr/bin/python
import socket
import select
import string
import sys
import json
from pprint import pprint

def prompt():
    sys.stdout.write("\nEnter actions like this: <action> <door number>\n+++Actions:+++\n\t(1) Choose door without powerup\n\t(2) Choose door with powerup 1\n\t(3) Choose door with powerup 2\nChoose an action and a door to open:")
    sys.stdout.flush()

if __name__ == "__main__":

    HOST = socket.gethostname()
    PORT = 5000
    BUFF_SIZE = 4096

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    
    try:
        s.connect((HOST, PORT))
    except:
        print ("Unable to connect")
        sys.exit()

    while True:
        connections = [sys.stdin, s]
        inputready, outputready, errorready = select.select(connections,[],[])

        for sock in inputready:
            if sock == s:
                data = sock.recv(BUFF_SIZE)
                pprint(data)
                #data_message = json.loads(data)
                if not data:
                    print ("Disconnected client")
                    sys.exit()
                else:
                    data = data.split("|")
                    sys.stdout.write(data[1])
                    if data[0].strip():
                        data[0] = int(data[0])                                    
                        if int(data[0]) != 0:
                            prompt()
            #sends event listener to server
            else:
                message = sys.stdin.readline()
                s.sendall(message)
               