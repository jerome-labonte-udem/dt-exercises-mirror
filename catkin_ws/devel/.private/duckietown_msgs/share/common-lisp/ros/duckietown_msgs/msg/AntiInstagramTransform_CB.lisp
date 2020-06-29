; Auto-generated. Do not edit!


(cl:in-package duckietown_msgs-msg)


;//! \htmlinclude AntiInstagramTransform_CB.msg.html

(cl:defclass <AntiInstagramTransform_CB> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (th
    :reader th
    :initarg :th
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 6 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass AntiInstagramTransform_CB (<AntiInstagramTransform_CB>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AntiInstagramTransform_CB>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AntiInstagramTransform_CB)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name duckietown_msgs-msg:<AntiInstagramTransform_CB> is deprecated: use duckietown_msgs-msg:AntiInstagramTransform_CB instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <AntiInstagramTransform_CB>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader duckietown_msgs-msg:header-val is deprecated.  Use duckietown_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'th-val :lambda-list '(m))
(cl:defmethod th-val ((m <AntiInstagramTransform_CB>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader duckietown_msgs-msg:th-val is deprecated.  Use duckietown_msgs-msg:th instead.")
  (th m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AntiInstagramTransform_CB>) ostream)
  "Serializes a message object of type '<AntiInstagramTransform_CB>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'th))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AntiInstagramTransform_CB>) istream)
  "Deserializes a message object of type '<AntiInstagramTransform_CB>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:setf (cl:slot-value msg 'th) (cl:make-array 6))
  (cl:let ((vals (cl:slot-value msg 'th)))
    (cl:dotimes (i 6)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AntiInstagramTransform_CB>)))
  "Returns string type for a message object of type '<AntiInstagramTransform_CB>"
  "duckietown_msgs/AntiInstagramTransform_CB")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AntiInstagramTransform_CB)))
  "Returns string type for a message object of type 'AntiInstagramTransform_CB"
  "duckietown_msgs/AntiInstagramTransform_CB")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AntiInstagramTransform_CB>)))
  "Returns md5sum for a message object of type '<AntiInstagramTransform_CB>"
  "ad95a08e9897d117b5fa255272552409")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AntiInstagramTransform_CB)))
  "Returns md5sum for a message object of type 'AntiInstagramTransform_CB"
  "ad95a08e9897d117b5fa255272552409")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AntiInstagramTransform_CB>)))
  "Returns full string definition for message of type '<AntiInstagramTransform_CB>"
  (cl:format cl:nil "Header header~%int16[6] th~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AntiInstagramTransform_CB)))
  "Returns full string definition for message of type 'AntiInstagramTransform_CB"
  (cl:format cl:nil "Header header~%int16[6] th~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AntiInstagramTransform_CB>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'th) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AntiInstagramTransform_CB>))
  "Converts a ROS message object to a list"
  (cl:list 'AntiInstagramTransform_CB
    (cl:cons ':header (header msg))
    (cl:cons ':th (th msg))
))
