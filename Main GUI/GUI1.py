# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:35:05 2022

@author: guypu
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from selenium import webdriver
import pafy
import functions
import cv2
import matplotlib.pyplot as plt
import threading
import queue
import flashdetect

class Root(tk.Tk):
    def __init__(self):
            super().__init__()
            
            self.parameter_list = functions.extract_Hyper_param("HyperParam.txt")
            
            self.title("The Anti Flasher")
            self.geometry("850x550")
            self.config(bg="#FFD133") # set background color of root window
             
            self.header = tk.Label(self, text="Anti-Flasher", bg="#FFB833", fg='white',relief='raised')

            self.header.config(font=("Font", 30)) # change font and size of label
            
            #self.image = tk.PhotoImage(file="logo.png")
         
            self.canvas = tk.Canvas(self, width = 170, height = 170)      
    
            self.img = tk.PhotoImage(file="logo.png")      
            self.canvas.create_image(1,1, anchor='nw', image=self.img) 


            self.Chrome_button = tk.Button(text="Open Chrome", command=self.Open_Chrome)
            self.Flag = 0
            self.Url_check = tk.Button(text="Check if video was uploaded", command=self.URLog )
            
            self.loadimage = tk.PhotoImage(file="RED_BUTTON.png")
            self.RED_Button = tk.Button(self, image=self.loadimage , command = self.Red_butt)

            self.Play_safeeee = tk.Button(text="Play_safe", command=self.Play_safe_func, bg="#3CFF33")
            self.messageframe = tk.LabelFrame(self,text='Messages', width = 300)

            self.console_text = tk.Message(self.messageframe, text="Welcome to the Anti flasher, first open browser", width = 250)
            #self.label_Consule = tk.Label(master=self.frameConsule, text="this is consule bitch")
            self.epileptic_list = ["black & white",
                              "color",
                              "full frame"]
            self.meth_var = tk.StringVar(self)
            self.meth_var.set("Epileptic Sensetivity")
            self.meth_entry = tk.OptionMenu(self, self.meth_var, *self.epileptic_list) 
            '''
            Figure format
            '''
            self.fig = Figure(figsize=(2.5,2.5) )
            self.Canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.result_butt = tk.Button(text="results", command=self.results, bg="#16f6fa")

            # self.check_Butt = tk.Button(text="chek mate", command=self.check_Butt, bg="#cc0099")
            
       
            self.header.pack(ipady=5, fill='x')
            self.messageframe.pack(side='right',fill='y') 
            self.messageframe.pack_propagate(0)
            self.console_text.pack() 
            self.console_text.pack_propagate(0)
            self.result_butt.pack(side='right')
            self.RED_Button.pack(side='right')
            self.meth_entry.pack(side='right')
            self.canvas.pack()            
            self.Chrome_button.pack()
            self.Url_check.pack()
            self.Play_safeeee.pack()
            # self.check_Butt.pack(side='right')
            self.Canvas.get_tk_widget().pack()  
            self.meth_entry.pack(side='right')
            
            '''
            functions in GUI
            '''
    def read_queue(self):
        """ Check for updated temp data"""
        try:
            temp = self.q.get_nowait()
            if temp:
                self.result1 = temp
                # print(self.result1)
                return
        except queue.Empty:
            pass
        # Schedule read_queue again in one second.
        self.after(0, self.read_queue)   
            
    def Open_Chrome(self):
        # "C:\BGU\Semester 8\Algo\Proj\chromedriver.exe"
        # "C:\Program Files\Google\Chrome\Application\chromedriver.exe"
        self.Driver = webdriver.Chrome(executable_path="C:\Program Files\Google\Chrome\Application\chromedriver.exe")
        self.Driver.get("https://www.google.com") #for now we need to keep in youtube
    
    def results(self):
       
         self.timeRGB = self.result1[0]         
         self.freqenciesFoundRGB = self.result1[1] 
         self.powerSpectrumRGB = self.result1[2]
         self.tsBW = self.result1[3]
         self.tsColor = self.result1[4]
         self.tsFull = self.result1[5]
         self.time_frames = self.result1[6]
         
         if self.meth_var.get() == "Epileptic Sensetivity" :    
            self.console_text.config(text="choose epileptic sesetivity:\n " )
         if self.meth_var.get() == self.epileptic_list[0]:   #Black & White     
            self.TS = self.tsBW               
         if self.meth_var.get() == self.epileptic_list[1]:   #Color    
            self.TS = self.tsColor             
         if self.meth_var.get() == self.epileptic_list[2]:         # Full Frame
            self.TS = self.tsFull
            

         self.seazure_per = len(self.TS)/len(self.timeRGB)*100
         self.console_text.config(text="The dangerous Timestamps for are:\n " + str(self.TS) +"\n"+
                                  "The percentage of harmful effects of this video is " +  str(int(self.seazure_per)) +'%')


         self.fig.clear()
         self.fig = plt.figure(figsize=[8, 6],clear=True)
         plt.pcolormesh(self.timeRGB, self.freqenciesFoundRGB, self.powerSpectrumRGB,cmap='coolwarm')
         plt.ylabel('Frequency [Hz]')
         plt.xlabel('Time [sec]')
         plt.title(self.Clip_Name + "spectogram")
         plt.close()
                
         self.Canvas.get_tk_widget().destroy()
         self.Canvas = FigureCanvasTkAgg(self.fig, master=self)
         self.Canvas.draw()
         self.Canvas.get_tk_widget().pack()
                   
  
    def URLog(self):
        File = open("Data\\URLog.txt","r+")
        self.Log = File.readlines() # this is a list pre-saved for checkingif url has been ulpoaded already
        
        self.CurrURL = self.Driver.current_url
        '''
        special naming 
        
        '''
        if "youtube" not in self.CurrURL:
            self.console_text.config(text="This is not YouTube")
            return
        
        UrlSplit = self.CurrURL.split("https://www.youtube.com/watch?v=")
        url11 = UrlSplit[1][0:11]    
        temp = url11
        
        #optimizing the URL name to save as a text
        for ch in ['\\','"','*','<','>','#','|','?',':','/',']','[']:
            if ch in temp:
                print(temp)
                temp = temp.replace(ch,"")
            
        
        '''
        a loop to search if the url is existed in URLog
        '''
        for i in range(len(self.Log)):
           self.Flag = 0 

           if  self.Log[i] == temp+'\n': #find the index where the the word is not equal
               self.console_text.config(text="this clip exist in our storage")
               self.Flag = 1
               
               break

        if  self.Log[len(self.Log)-1] == temp: #last one has no "\n"

            self.console_text.config(text="this clip exist in our storage")
            self.Flag = 1
             
        File.close()
        

        # in case we have "youtube" in our URL and we did not downloaded the video
        if "youtube" in self.CurrURL and not self.Flag:
            self.clip = pafy.new(self.CurrURL) 
            Download_thread = threading.Thread(target=lambda: functions.Download_clip(self.clip))
            Download_thread.start()       
            self.console_text.config(text="This clip IS NOT exist in our storage \n Downloading")
            
        # in case we have "youtube" in out URL and we  downloaded the video
        if "youtube" in self.CurrURL and  self.Flag:
            self.clip = pafy.new(self.CurrURL) 

    def Red_butt(self):
       
        if self.meth_var.get() == "Epileptic Sensetivity" :    
                self.console_text.config(text="Pleas choose epileptic sesetivity:\n Then start " )
                return
        if self.Flag==0:

            ''' 
            1. first update the URLog txt file
            '''
            self.console_text.config(text="Adding clip to storage")
            File = open("Data\\URLog.txt","r+")
            Log = File.readlines()# SAVING THE CURRENT FILE
            File.seek(0) #Deleting the file contenet
            File.truncate() 
            '''
            special naming 
            
            '''
            UrlSplit = self.CurrURL.split("https://www.youtube.com/watch?v=")
            url11 = UrlSplit[1][0:11]    
            temp = url11
            
            #optimizing the URL name to save as a text
            for ch in ['\\','"','*','<','>','#','|','?',':','/','[',']']:
                if ch in temp:
                    print(temp)
                    temp = temp.replace(ch,"")
                    
            Log.append(temp+'\n')
            Log.sort()
            File.writelines(Log)         
            File.close()
            
            
            '''
            runing pack 1 requires downloaded clip this here to break if download hasnt end
            '''
            # self.URLog()# Update the flag
            self.Clip_Name = self.clip.title   
            self.cap = cv2.VideoCapture(self.Clip_Name + '.mp4')
            
            if not self.cap.isOpened():
                self.console_text.config(text="Wait for download to end")
                return
                    
            if "youtube" in self.CurrURL: #double check... there is no such thing 2 safe          
      
                self.q = queue.Queue()
                self.file_Path = self.Clip_Name + '.mp4'
                threading.Thread(target=flashdetect.Start_detect, args =  (self.file_Path ,self.q, self)).start()
                self.after(0, self.read_queue)
                
                
        else: # we have download the video earlier   
        
            self.tsBW, self.tsColor, self.tsFull, self.timeRGB ,self.freqenciesFoundRGB, self.time_frames, self.powerSpectrumRGB = functions.Load_Data(self.CurrURL)
            
            '''
            creating the spectogram based on pack 1
            '''

            self.fig.clear()
            self.fig = plt.figure(figsize=[8, 6],clear=True)
            plt.pcolormesh(self.timeRGB[0], self.freqenciesFoundRGB[0], self.powerSpectrumRGB,cmap='coolwarm')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title("specgram")
            plt.close()

            self.Canvas.get_tk_widget().destroy()
            self.Canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.Canvas.draw()
            self.Canvas.get_tk_widget().pack()
            
            if self.meth_var.get() == "Epileptic Sensetivity" :    
                self.console_text.config(text="choose epileptic sesetivity:\n " )
            if self.meth_var.get() == self.epileptic_list[0]:   #Black & White     
                self.TS = self.tsBW               
            if self.meth_var.get() == self.epileptic_list[1]:   #Color    
                self.TS = self.tsColor             
            if self.meth_var.get() == self.epileptic_list[2]:         # Full Frame
                self.TS = self.tsFull
                

            self.seazure_per = len(self.TS[0])/len(self.timeRGB[0])*100
            self.console_text.config(text="The dangerous Timestamps for are:\n " + str(self.TS[0]) +"\n"+
                                         "The percentage of harmful effects of this video is " +  str(int(self.seazure_per)) +'%')

  

    def Play_safe_func(self):
        
        if self.Flag == 1:
            self.console_text.config(text="Clip not exist in directory")
            return
        
        if self.time_frames[0]==0:
            self.fps = round(1/self.time_frames[1]) 
        else: 
            self.fps = round(1/self.time_frames[0]) 
                             
        self.alert_image = functions.cv2.imread(filename="warning.png")
            
      
        functions.fix_frames(self.Clip_Name + '.mp4',self.TS,self.fps,self.time_frames.T,self.alert_image,self.parameter_list[1][12])

if __name__ == "__main__":
    
    root = Root()
    root.mainloop()
