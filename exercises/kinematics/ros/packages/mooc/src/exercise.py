#!/usr/bin/env python
# coding: utf-8

# # Canny Edge detector
# 
# Some notes on the topic of the exercise <br><br>
# 
# 
# Link to the videos related to the exercise <br>
# 
# 

# ### Task 1
# <br>
# Write the Canny edge detector function in the cell below. 
# 
# Do not change the name of the function.
# 

# In[ ]:


#
#   import libraries
#

import cv2


# In[ ]:


#
#   Canny edge function the student needs to write
# 

#
# WARING: Do not change the name of the function
#
# You can define all the other functions you need
#



def CannyF(input_image):
    # this function has to return a b/w image 
    # the image is a bgr image ...
    # some other instructions ...
    
    lowThreshold=75
    highThreshold=150
    return cv2.Canny(input_image, lowThreshold, highThreshold)


    

