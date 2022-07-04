import matplotlib
matplotlib.use('TkAgg')
import functions
import cv2


def Start_detect(file_Path,q,self):
    self.console_text.config(text="Flash detection started\n " )
    parameter_list = functions.extract_Hyper_param("HyperParam.txt")
    cap = cv2.VideoCapture(file_Path)
    # cap, flag, percent,Percentage ,dim_for_CLRGB,Ms_to_sec
    
    frames, frameCount, frameWidth, frameHeight, fps, time_frames = functions.read_vid_to_frames(
        cap, False,parameter_list[1][7],parameter_list[1][5],int(parameter_list[1][3]),parameter_list[1][11])
                
    frames_r, frames_b, frames_g, frames_r_minus_b = functions.extract_color(
        frames, int(parameter_list[1][0]), int(parameter_list[1][1]), int(parameter_list[1][2]))
                
    ''' 
    last so far
    '''                
    cluster_frameRGB = functions.cluster_frames_RGB(
        frames,frameCount,frameHeight,frameWidth,int(parameter_list[1][8]),int(parameter_list[1][3]))
                
              
    boolvec,time_stamps_4_by_4,mainTS_RGB,time = functions.cluster_time_stempsRGB(
        cluster_frameRGB,frameCount,fps,int(parameter_list[1][9]),self.Clip_Name)
                
    ''' 
    second best so far
    '''
                
    time_stamps_Three_channel = []
    
    Blue_TS,timee = functions.cluster_time_stemps(
        cluster_frameRGB[0,:,:,:],frameCount,fps,int(parameter_list[1][9]))
    time_stamps_Three_channel.append(Blue_TS)
                
    Red_TS,timee = functions.cluster_time_stemps(
        cluster_frameRGB[2,:,:,:],frameCount,fps,int(parameter_list[1][9]))
    time_stamps_Three_channel.append(Red_TS)
                       
    Green_TS,timee = functions.cluster_time_stemps(
        cluster_frameRGB[1,:,:,:],frameCount,fps,int(parameter_list[1][9]))
    time_stamps_Three_channel.append(Green_TS)   
                
    mainTS_cluster_each_colour_and_find_best = functions.Take_best_TS(time_stamps_Three_channel,timee)
                
    ''' 
    best so far
    '''
    cluster_frameB = functions.dense_cluster_frames(frames_b, frameCount)
    cluster_frameR = functions.dense_cluster_frames(frames_r, frameCount)
    cluster_frameG = functions.dense_cluster_frames(frames_g, frameCount)
    cluster_RGB_mat = functions.np.empty(
        (int(parameter_list[1][3]), frameCount), functions.np.dtype('int32'))
    cluster_RGB_mat[0, :] = cluster_frameR
    cluster_RGB_mat[1, :] = cluster_frameG
    cluster_RGB_mat[2, :] = cluster_frameB
    cluster_RGB_vec = functions.np.mean(cluster_RGB_mat, axis=0)
                              
    powerSpectrumRGB, freqenciesFoundRGB, timeRGB = functions.Spectrograma_with_print_option(
            cluster_RGB_vec, frameCount, fps,  self.Clip_Name, 0)
    # saving results:
                
    RGB_time_with_seazure_prob, time_stampsRGB = functions.find_time_stamps(
                powerSpectrumRGB, freqenciesFoundRGB, timeRGB, fps)
    
    functions.Save_Data(self.Demo_URL,timeRGB, freqenciesFoundRGB, powerSpectrumRGB,mainTS_RGB,mainTS_cluster_each_colour_and_find_best,time_stampsRGB,time_frames )
    self.console_text.config(text="done\n " )
    q.put([timeRGB, freqenciesFoundRGB, powerSpectrumRGB,mainTS_RGB,mainTS_cluster_each_colour_and_find_best,time_stampsRGB,time_frames]) 
                