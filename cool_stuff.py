import webbrowser
import cv2

def open_google_image(word):

	url="http://images.google.com/search?q="+word+"&tbm=isch"
	webbrowser.open(url, new=0, autoraise=True)

def display_image(path,delay=0):
	img = cv2.imread(path,0)
	winname = "ArielIvanDiaz.net"
	cv2.namedWindow(winname)  
	cv2.moveWindow(winname, 1600,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey(delay)
	cv2.destroyAllWindows()
