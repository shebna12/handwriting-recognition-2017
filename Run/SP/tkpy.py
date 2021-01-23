from tkinter import *
from tkinter import ttk
import socket

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

root = Tk()

# frame = Frame(root)
# labelText = StringVar()
# label = Label(frame,textvariable = labelText)
# labelText.set("Welcome to the magical world of SM!*supermalls*")
# # button = Button 
# button = Button(frame, text = "Start Game")
# label.pack()
# button.pack()
# frame.pack()

#Geometry manager
#each cell can hold one widget but more cells can too

Label(root, text="First Name").grid(row=0, sticky=W, padx = 4)
Entry(root).grid(row=0, column=1, sticky=E, pady=4)

Label(root, text="Last Name").grid(row=1, sticky=W, padx = 4)
Entry(root).grid(row=1, column=1, sticky=E, pady=4)

Button(root, text="submit").grid(row=3)

#row = 0 top
#sticky how the widgets gonna expand to the west
#N, NE, NW, E, W
#padding
root.mainloop()