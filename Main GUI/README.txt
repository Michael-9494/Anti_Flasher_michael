our project

in order to run "GUI1" there is a need to download 3 modules and to change one of them MANUALY 

the 3 modules are:
selenium
pafy
youtube-dl

in this algorithem, there is a use of a browser. in order to run a browser via python with selenium module you'll need to download a specific webdriver
selenium provides those drivers in the next URL 

https://www.selenium.dev/documentation/webdriver/getting_started/install_drivers/

please make sure to download the same driver as your current driver you are using
place the driver in a specific place and log its path.


then you'll need to debug a python script that is in pafy:
1. go to file " backend_youtube_dl.py " that is in pafy file.    
	# in my computer the path for this is : C:\Users\guypu\anaconda3\envs\Algo\Lib\site-packages\pafy
	# NOTE: this is the path on my computer the location of each package differ between different user but the idea is the same

2. change lines 53 and 54 from :
        53.		self._likes = self._ydl_info['like_count']
        54.		self._dislikes = self._ydl_info['dislike_count']

       to:
        53.		self._likes = self._ydl_info.get('like_count', 0)
        54.		self._dislikes = self._ydl_info.get('dislike_count', 0)

3. save changes
   
for more information: https://github.com/mps-youtube/pafy/pull/288 

