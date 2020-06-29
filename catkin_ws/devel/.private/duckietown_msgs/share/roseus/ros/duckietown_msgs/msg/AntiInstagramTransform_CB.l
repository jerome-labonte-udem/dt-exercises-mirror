;; Auto-generated. Do not edit!


(when (boundp 'duckietown_msgs::AntiInstagramTransform_CB)
  (if (not (find-package "DUCKIETOWN_MSGS"))
    (make-package "DUCKIETOWN_MSGS"))
  (shadow 'AntiInstagramTransform_CB (find-package "DUCKIETOWN_MSGS")))
(unless (find-package "DUCKIETOWN_MSGS::ANTIINSTAGRAMTRANSFORM_CB")
  (make-package "DUCKIETOWN_MSGS::ANTIINSTAGRAMTRANSFORM_CB"))

(in-package "ROS")
;;//! \htmlinclude AntiInstagramTransform_CB.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass duckietown_msgs::AntiInstagramTransform_CB
  :super ros::object
  :slots (_header _th ))

(defmethod duckietown_msgs::AntiInstagramTransform_CB
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:th __th) (make-array 6 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _header __header)
   (setq _th __th)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:th
   (&optional __th)
   (if __th (setq _th __th)) _th)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int16[6] _th
    (* 2    6)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int16[6] _th
     (dotimes (i 6)
       (write-word (elt _th i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int16[6] _th
   (dotimes (i (length _th))
     (setf (elt _th i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;;
   self)
  )

(setf (get duckietown_msgs::AntiInstagramTransform_CB :md5sum-) "ad95a08e9897d117b5fa255272552409")
(setf (get duckietown_msgs::AntiInstagramTransform_CB :datatype-) "duckietown_msgs/AntiInstagramTransform_CB")
(setf (get duckietown_msgs::AntiInstagramTransform_CB :definition-)
      "Header header
int16[6] th

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



(provide :duckietown_msgs/AntiInstagramTransform_CB "ad95a08e9897d117b5fa255272552409")


