print("-----------WORD COUNT PROGRAM------------\n")

words_summary = {}


#Select a file name 
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_name = filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("all files","*.*"), ("jpeg files","*.jpg")))


if file_name == None or file_name == "":
    print("No File selected")
    exit(0)

    
#Opening a file
with open(file_name, 'r') as f:
    
    line = f.readline()
    while line != "":
        for word in line.split():
            if word in words_summary:
                words_summary[word] += 1
            else:
                words_summary[word] = 1
        
        line = f.readline()

#Writing to a file
with open('wordcount_output.txt', 'w') as f:
    for key, value in words_summary.items():
        f.write(key + ": " + str(value) + "\n")
        print(key + ": " + str(value) + "\n")
