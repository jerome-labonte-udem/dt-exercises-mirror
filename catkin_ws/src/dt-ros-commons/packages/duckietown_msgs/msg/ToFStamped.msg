Header header

# There is no error. `distance` is valid.
int8 NO_ERROR = 0
# Target is too close to the sensor
int8 ERROR_NEAR = 1
# Target is too far from the sensor (> 2047mm)
int8 ERROR_FAR = 2
# Other general error
int8 ERROR_OTHER = 3

int8 error       # One of NO_ERROR, ERROR_NEAR, ERROR_FAR, or ERROR_OTHER
int16 distance   # Distance in mm. Only valid if error == NO_ERROR
int16 confidence # An abstract "confidence" measurement that is not well-defined in the RFD77402 datasheet