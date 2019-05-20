

# -*- coding:utf-8 -*-
# Imports
import sys
import os
from cv2 import *
from tkinter import * 
from tkinter import Tk, BOTH,  Menu, Canvas
from PIL import Image, ImageTk
from tkinter import filedialog
import testing

imgname = "l.jpeg"
width = 500
height = 800




# Function definitions
def deleteImage(canvas):
    canvas.delete("all")
    return

def quitProgram():
    gui.destroy()
# Main window
gui = Tk()
lab = Label(gui, text="WBC CLASSIFICATION",fg="red",justify='center')
lab.config(font=("Courier", 34 , 'bold'))
lab.config(bg="black")
lab.pack()

logo1 = PhotoImage(file="WBC.png")
w11 = Label(gui, image=logo1)
w11.image = logo1
w11.pack(side="right")
w11.place(x=600,y=250)
# BROWSE BUTTON CODE
def browsefunc():
 gui.configure(background='lightblue')
 global imgname
 imgname = filedialog.askopenfilename()
 pathlabel.config(text=imgname,fg = "red",
		 font = "Times")
pathlabel = Label()
pathlabel.pack(side=BOTTOM)
# Inside the main gui window
#Creating an object containing an image


#RUNNING PROGRAM
#def run():
#trafficcopy.main(imgname)
#import trafficcopy

def count():
   trafficcopy.main(imgname)
   s=trafficcopy.total_cars
   w = Label(gui, text="vehicle count is "+str(s),fg="blue")
   w.config(font=("Courier", 24,'bold'))
   w.config(bg="lightyellow")
   w.pack()
   w.place(x=500,y=600)
   sec=(trafficcopy.total_cars/2)*4
   
      #v = Label(gui, text=sec)
      #v.pack()
   def countdown(count):
      # change text in label        
      label['text'] = count

      if count > 0:
         # call countdown again after 1000ms (1s)
         gui.after(1000, countdown, count-1)

   label = Label(gui,fg="red")
   label.place(x=400, y=405)
   label.config(font=("Courier", 44,'bold'))
   label.config(bg="black")

   # call countdown first time    
   countdown(int(sec))


   logo = PhotoImage(file="sig.gif")
   w1 = Label(gui, image=logo)
   w1.image = logo
   w1.pack(side="right")
   w1.place(x=600,y=250)
   explanation = "Traffic Signal"
   w2 = Label(gui, 
               justify=LEFT,
               padx = 10, 
               text=explanation,fg="green")
   w2.config(font=("Courier", 44,'bold'))
   w2.config(bg="lightblue")
   w2.pack()
   w2.place(x=400,y=100)
   secc=sec*1000
   sec1=int(secc)
   gui.after(sec1, w1.destroy) # label as argument for destroy_widget
   w.after(sec1, lambda: w.destroy() )

def exits(): 
    gui.destroy()

def settings():
    window = Toplevel(gui)    
    gui.title("Settings")
    a = Label(gui ,text="First Name").grid(row=0,column = 0)
    E = Entry(gui).grid(row=0,column=1)

menu = Menu(gui)
gui.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label="Menu", menu=filemenu)
filemenu.add_command(label="Browse", command=browsefunc)
filemenu.add_command(label="Proceed", command=count)
#filemenu.add_command(label="Signal", command=)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=gui.quit)



gui.title("TRAFFIC CONTROL")
gui.attributes("-fullscreen", True)
gui.mainloop()
