#Add the shield code
#REname sowrd as item 1 and shield as item 2
#Checking mechanism for game: Is it game over? 
#


from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageSequence, ImageTk
import socket
import select


# class FullScreenApp(object):
#     def __init__(self, master, **kwargs):
#         self.master=master
#         pad=3
#         self._geom='200x200+0+0'
#         master.geometry("{0}x{1}+0+0".format(
#             master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
#         master.bind('<Escape>',self.toggle_geom)            
#     def toggle_geom(self,event):
#         geom=self.master.winfo_geometry()
#         print(geom,self._geom)
#         self.master.geometry(self._geom)
#         self._geom=geom




class AnimatedGif(object):
    """ Animated GIF Image Container. """
    def __init__(self, image_file_path):
        self.image_file_path = image_file_path
        self._load()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, frame_num):
        return self._frames[frame_num]

    def _load(self):
        """ Read in all the frames of a gif image. """
        self._frames = []
        img = Image.open(self.image_file_path)
        for frame in ImageSequence.Iterator(img):
            photo = ImageTk.PhotoImage(frame)
            photo.delay = frame.info['duration'] * 10  # add attribute
            self._frames.append(photo)

def update_label_image(label, anigif, frame_num):
    """ Change label image to given frame number of AnimatedGif. """
    global cancel_id
    frame = anigif[frame_num]
    label.configure(image=frame)
    frame_num = (frame_num+1) % len(anigif)  # next frame number
    cancel_id = second_frame.after(
                    frame.delay, update_label_image, label, anigif, frame_num)

def enable_animation(frame,anigif,label):
    """ Start animation of label image. """
    global cancel_id
    cancel_id = frame.after(
                    anigif[0].delay, update_label_image, label, anigif, 0)

def cancel_animation():
    """ Stop animation of label image. """
    global cancel_id
    if cancel_id is not None:
        second_frame.after_cancel(cancel_id)
        cancel_id = None





def raise_frame(frame):
    frame.tkraise()
def show_about(event=None):
		messagebox.showwarning(
            "About",
            "This program was made by Team DorMe-1"
		)

def decrease_sword_turns():
    sword_num_turns = sword_turns.get() - 1
    sword_turns.set(sword_num_turns) 
    sword_label.config(text = str(sword_num_turns_text))
    print("sword num turns: ", sword_turns.get())

    # sword_label.configure(text=str(sword_turns.get()))

def sel():
   try:
       if((var.get() != 0) and (door_var.get() != 0)):
        door_val = door_var.get()
        print("var: ",sword_turns.get())
        if(var.get() == 1):
            if(sword_turns.get() >= 0):
                selection = "You selected the door " + str(door_var.get()) + " and use item 1. "
                pauper_item = 1 #Use item variable as a means to send data to server
            else:
                selection = "You ran out of item 1! Choose another item or use nothing."        
        elif(var.get() == 2):
            if(shield_turns.get() >=0):
                selection = "You selected the door " + str(door_var.get()) + " and use item 2. "
                pauper_item = 2
            else:
                selection = "You ran out of item 2! Choose another item or use nothing." 
        elif(var.get() == 3):
            selection = "You selected the door " + str(door_var.get()) + " and use nothing. "
            pauper_item = 3
        
        pmessage_label.config(text = selection)
   except UnboundLocalError:
        pass

def sendData():
    if(var.get() != 0) and (door_var.get() !=0):
        # print("Na send ko na po ang data.")

        #---SEND ITEM DATA
        if(var.get() == 1 and sword_turns.get() != 0): #Decrease item 1 quantity
            sword_num_turns = sword_turns.get() - 1
            sword_turns.set(sword_num_turns) 
            sword_label.config(text = str(sword_num_turns))
            #----SEND DOOR DATA
        elif(var.get() == 2 and shield_turns.get() !=0): #Decrease item 2 quantitiy
            shield_num_turns = shield_turns.get() - 1
            shield_turns.set(shield_num_turns) 
            shield_label.config(text = str(shield_num_turns))
            #----SEND DOOR DATA

        elif(var.get() == 3):
            #----SEND DOOR DATA
            pass


#---START---#
root = Tk()
# root.geometry("300x300").
# root.grid_propagate(False)
width, height = root.winfo_screenwidth(), root.winfo_screenheight()

root.geometry('%dx%d+0+0' % (width,height))
# ttk.Style().theme_use('clam')
first_frame = Frame(root)
second_frame = Frame(root)
sword_turns = IntVar()
sword_turns.set(2)
shield_turns = IntVar()
shield_turns.set(2)
bomb_turns = IntVar()
bomb_turns.set(2)
item_remove_turns = IntVar()
item_remove_turns.set(2)










#-----Menu----#

the_menu = Menu(root)
file_menu = Menu(the_menu, tearoff=0)
file_menu.add_command(label="About",command=show_about)
the_menu.add_cascade(label="Help",menu=file_menu)
root.config(menu=the_menu)

for frame in (first_frame,second_frame):
    frame.grid(row=0, column=0, sticky='news')

#---buttons---#

labelText = StringVar()

start_label = Label(first_frame, textvariable = labelText).grid(row=0,column=0,columnspan=3)
start_button = Button(first_frame, text="Start Game",command=lambda:raise_frame(second_frame)).grid(row=1,column=1)
help_button = Button(first_frame, text="Help").grid(row=2,column=1)
exit_button = Button(first_frame, text="Exit").grid(row=3,column=1)
labelText.set("Welcome to Pauper VS Wizard!")




#--------------IMAGES------------#
background_img = Image.open('FinalProjectPictures/floor.png')
tk_img = ImageTk.PhotoImage(background_img)
background_panel = Label(second_frame,image=tk_img).grid(row=0,rowspan=8,column=0,columnspan=8,sticky=W)
player=2
if(player ==1 ):
    sword_label = Label(second_frame,text = sword_turns.get())
    sword_label.grid(row=6,column=2,sticky=S)

    shield_label = Label(second_frame,text = shield_turns.get())
    shield_label.grid(row=6,column=3,sticky=S)

    sword_img = Image.open('FinalProjectPictures/sword.png')
    tk_img_sword = ImageTk.PhotoImage(sword_img)
    # sword_button = Button(second_frame,image=tk_img_sword,command=decrease_sword_turns)
    # sword_button = Button(second_frame,image=tk_img_sword)
    # sword_button.bind("<Button-1>",decrease_sword_turns())
    sword_panel = Button(second_frame,image=tk_img_sword).grid(row=6,column=2)


    shield_img = Image.open('FinalProjectPictures/shield.png')
    tk_img_shield = ImageTk.PhotoImage(shield_img)
    # shield_button = Button(second_frame,image=tk_img_sword,command=decrease_sword_turns)
    # shield_button = Button(second_frame,image=tk_img_sword)
    shield_panel = Button(second_frame,image=tk_img_shield).grid(row=6,column=3)

    cancel_id = None
    image_file_path = 'FinalProjectPictures/pauper.gif'
    anigif = AnimatedGif(image_file_path)
    pauper_label = Label(second_frame,image=anigif[0])  # display first frame initially
    pauper_label.grid(row=6,rowspan=2,column=0)

    image_file_path_wizard = 'FinalProjectPictures/wizard.gif'
    anigif_wizard = AnimatedGif(image_file_path_wizard)
    wizard_label = Label(second_frame,image=anigif_wizard[0])  # display first frame initially
    wizard_label.grid(row=0,rowspan=2,column=0)
    enable_animation(second_frame,anigif_wizard,wizard_label)
elif(player == 2):
    bomb_label = Label(second_frame,text = bomb_turns.get())
    bomb_label.grid(row=6,column=2,sticky=S)

    item_remove_label = Label(second_frame,text = item_remove_turns.get())
    item_remove_label.grid(row=6,column=3,sticky=S)

    bomb_img = Image.open('FinalProjectPictures/bomb_up.png')
    tk_img_bomb = ImageTk.PhotoImage(bomb_img)
    # sword_button = Button(second_frame,image=tk_img_sword,command=decrease_sword_turns)
    # sword_button = Button(second_frame,image=tk_img_sword)
    # sword_button.bind("<Button-1>",decrease_sword_turns())
    bomb_panel = Button(second_frame,image=tk_img_bomb).grid(row=6,column=2)


    item_remove_img = Image.open('FinalProjectPictures/item_remove.png')
    tk_img_item_remove = ImageTk.PhotoImage(item_remove_img)
    # shield_button = Button(second_frame,image=tk_img_sword,command=decrease_sword_turns)
    # shield_button = Button(second_frame,image=tk_img_sword)
    item_remove_panel = Button(second_frame,image=tk_img_item_remove).grid(row=6,column=3)

    cancel_id = None
    image_file_path = 'FinalProjectPictures/wizard.gif'
    anigif = AnimatedGif(image_file_path)
    wizard_label = Label(second_frame,image=anigif[0])  # display first frame initially
    wizard_label.grid(row=6,rowspan=2,column=0)
    enable_animation(second_frame,anigif,wizard_label)

    image_file_path_pauper = 'FinalProjectPictures/pauper.gif'
    anigif_pauper = AnimatedGif(image_file_path_pauper)
    pauper_label = Label(second_frame,image=anigif_pauper[0])  # display first frame initially
    pauper_label.grid(row=0,rowspan=2,column=0)
    enable_animation(second_frame,anigif_pauper,pauper_label)

image_file_path_door1 = 'FinalProjectPictures/door_1.gif'
anigif_door1 = AnimatedGif(image_file_path_door1)
door1_label = Label(second_frame,image=anigif_door1[0])  # display first frame initially
door1_label.grid(row=2,rowspan=4,column=1,columnspan=2)
enable_animation(second_frame,anigif_door1,door1_label)

image_file_path_door2 = 'FinalProjectPictures/door_2.gif'
anigif_door2 = AnimatedGif(image_file_path_door2)
door2_label = Label(second_frame,image=anigif_door2[0])  # display first frame initially
door2_label.grid(row=2,rowspan=4,column=3,columnspan=2)
enable_animation(second_frame,anigif_door2,door2_label)

image_file_path_door3 = 'FinalProjectPictures/door_3.gif'
anigif_door3 = AnimatedGif(image_file_path_door3)
door3_label = Label(second_frame,image=anigif_door3[0])  # display first frame initially
door3_label.grid(row=2,rowspan=4,column=5,columnspan=2)
enable_animation(second_frame,anigif_door3,door3_label)


#---Label for Item PowerUps---#


# start_animation = Button(second_frame, text="start animation", command=enable_animation)
# start_animation.pack()
# stop_animation = Button(second_frame, text="stop animation", command=cancel_animation)
# stop_animation.pack()
# exit_program = Button(second_frame, text="exit", command=root.quit)
# exit_program.pack()
# root.geometry("250x125+100+100")


#----INPUT BOX-----#

# input_box = Entry(second_frame).grid(row=8,column=0,columnspan=8,sticky=W)

var = IntVar()
door_var = IntVar()
R1 = Radiobutton(second_frame, text="Open door 1", variable=door_var, value=1,
                  command=sel)
R1.grid(row=9,column=0,columnspan=4,sticky=W)

R2 = Radiobutton(second_frame, text="Open door 2", variable=door_var, value=2,
                  command=sel)
R2.grid(row=10,column=0,columnspan=4,sticky=W)

R3 = Radiobutton(second_frame, text="Open door 3", variable=door_var, value=3,
                  command=sel)
R3.grid(row=11,column=0,columnspan=4,sticky=W)

I1 = Radiobutton(second_frame, text="Use item 1", variable=var, value="1",
                  command=sel)
I2 = Radiobutton(second_frame, text="Use item 2", variable=var, value="2",
                  command=sel)
I3 = Radiobutton(second_frame, text="Use nothing", variable=var, value="3",
                  command=sel)
I1.grid(row=9,column=5,columnspan=4,sticky=W)
I2.grid(row=10,column=5,columnspan=4,sticky=W)
I3.grid(row=11,column=5,columnspan=4,sticky=W)
# I4.grid(row=12,column=5,columnspan=4,sticky=W)
pmessage_label = Label(second_frame)
pmessage_label.grid(row=8,column=0,columnspan=4,sticky=W)



send_button = Button(second_frame,text="SUBMIT",command=sendData).grid(row=8,column=6,columnspan=2)
# opt1 = Label(second_frame, text = "Enter 1 to open a door. \n").grid(row=9,column=0,columnspan=4,sticky=W)
# opt2 = Label(second_frame,text = "Enter 2 to use item1.\n" ).grid(row=10,column=0,columnspan=4,sticky=W)
# opt3 = Label(second_frame,text = "Enter 3 to use item2.\n").grid(row=11,column=0,columnspan=4,sticky=W)



# start_label.grid(row = 0,column = 2,sticky = N, pady=4)
# start_button.grid(row = 1,column = 2, sticky=N, pady=5)
# help_button.grid(row = 2,column = 2, sticky=N, pady=5)
# exit_button.grid(row = 3,column = 2, sticky=N, pady=5)




raise_frame(first_frame)
# app=FullScreenApp(root)
root.mainloop()
