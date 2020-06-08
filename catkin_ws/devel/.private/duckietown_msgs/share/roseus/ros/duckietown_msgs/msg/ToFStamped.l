;; Auto-generated. Do not edit!


(when (boundp 'duckietown_msgs::ToFStamped)
  (if (not (find-package "DUCKIETOWN_MSGS"))
    (make-package "DUCKIETOWN_MSGS"))
  (shadow 'ToFStamped (find-package "DUCKIETOWN_MSGS")))
(unless (find-package "DUCKIETOWN_MSGS::TOFSTAMPED")
  (make-package "DUCKIETOWN_MSGS::TOFSTAMPED"))

(in-package "ROS")
;;//! \htmlinclude ToFStamped.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*NO_ERROR*" (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(shadow '*NO_ERROR* (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(defconstant duckietown_msgs::ToFStamped::*NO_ERROR* 0)
(intern "*ERROR_NEAR*" (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(shadow '*ERROR_NEAR* (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(defconstant duckietown_msgs::ToFStamped::*ERROR_NEAR* 1)
(intern "*ERROR_FAR*" (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(shadow '*ERROR_FAR* (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(defconstant duckietown_msgs::ToFStamped::*ERROR_FAR* 2)
(intern "*ERROR_OTHER*" (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(shadow '*ERROR_OTHER* (find-package "DUCKIETOWN_MSGS::TOFSTAMPED"))
(defconstant duckietown_msgs::ToFStamped::*ERROR_OTHER* 3)
(defclass duckietown_msgs::ToFStamped
  :super ros::object
  :slots (_header _error _distance _confidence ))

(defmethod duckietown_msgs::ToFStamped
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:error __error) 0)
    ((:distance __distance) 0)
    ((:confidence __confidence) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _error (round __error))
   (setq _distance (round __distance))
   (setq _confidence (round __confidence))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:error
   (&optional __error)
   (if __error (setq _error __error)) _error)
  (:distance
   (&optional __distance)
   (if __distance (setq _distance __distance)) _distance)
  (:confidence
   (&optional __confidence)
   (if __confidence (setq _confidence __confidence)) _confidence)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int8 _error
    1
    ;; int16 _distance
    2
    ;; int16 _confidence
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int8 _error
       (write-byte _error s)
     ;; int16 _distance
       (write-word _distance s)
     ;; int16 _confidence
       (write-word _confidence s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int8 _error
     (setq _error (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _error 127) (setq _error (- _error 256)))
   ;; int16 _distance
     (setq _distance (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _confidence
     (setq _confidence (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get duckietown_msgs::ToFStamped :md5sum-) "803258046ff4b4331a4d8f4f3f180cc9")
(setf (get duckietown_msgs::ToFStamped :datatype-) "duckietown_msgs/ToFStamped")
(setf (get duckietown_msgs::ToFStamped :definition-)
      "Header header

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
int16 confidence # An abstract \"confidence\" measurement that is not well-defined in the RFD77402 datasheet
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :duckietown_msgs/ToFStamped "803258046ff4b4331a4d8f4f3f180cc9")


