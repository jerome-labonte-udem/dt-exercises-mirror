; Auto-generated. Do not edit!


(cl:in-package duckietown_msgs-msg)


;//! \htmlinclude ToFStamped.msg.html

(cl:defclass <ToFStamped> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (error
    :reader error
    :initarg :error
    :type cl:fixnum
    :initform 0)
   (distance
    :reader distance
    :initarg :distance
    :type cl:fixnum
    :initform 0)
   (confidence
    :reader confidence
    :initarg :confidence
    :type cl:fixnum
    :initform 0))
)

(cl:defclass ToFStamped (<ToFStamped>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ToFStamped>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ToFStamped)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name duckietown_msgs-msg:<ToFStamped> is deprecated: use duckietown_msgs-msg:ToFStamped instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ToFStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader duckietown_msgs-msg:header-val is deprecated.  Use duckietown_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <ToFStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader duckietown_msgs-msg:error-val is deprecated.  Use duckietown_msgs-msg:error instead.")
  (error m))

(cl:ensure-generic-function 'distance-val :lambda-list '(m))
(cl:defmethod distance-val ((m <ToFStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader duckietown_msgs-msg:distance-val is deprecated.  Use duckietown_msgs-msg:distance instead.")
  (distance m))

(cl:ensure-generic-function 'confidence-val :lambda-list '(m))
(cl:defmethod confidence-val ((m <ToFStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader duckietown_msgs-msg:confidence-val is deprecated.  Use duckietown_msgs-msg:confidence instead.")
  (confidence m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<ToFStamped>)))
    "Constants for message type '<ToFStamped>"
  '((:NO_ERROR . 0)
    (:ERROR_NEAR . 1)
    (:ERROR_FAR . 2)
    (:ERROR_OTHER . 3))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'ToFStamped)))
    "Constants for message type 'ToFStamped"
  '((:NO_ERROR . 0)
    (:ERROR_NEAR . 1)
    (:ERROR_FAR . 2)
    (:ERROR_OTHER . 3))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ToFStamped>) ostream)
  "Serializes a message object of type '<ToFStamped>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let* ((signed (cl:slot-value msg 'error)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'distance)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'confidence)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ToFStamped>) istream)
  "Deserializes a message object of type '<ToFStamped>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'error) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'distance) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'confidence) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ToFStamped>)))
  "Returns string type for a message object of type '<ToFStamped>"
  "duckietown_msgs/ToFStamped")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ToFStamped)))
  "Returns string type for a message object of type 'ToFStamped"
  "duckietown_msgs/ToFStamped")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ToFStamped>)))
  "Returns md5sum for a message object of type '<ToFStamped>"
  "803258046ff4b4331a4d8f4f3f180cc9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ToFStamped)))
  "Returns md5sum for a message object of type 'ToFStamped"
  "803258046ff4b4331a4d8f4f3f180cc9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ToFStamped>)))
  "Returns full string definition for message of type '<ToFStamped>"
  (cl:format cl:nil "Header header~%~%# There is no error. `distance` is valid.~%int8 NO_ERROR = 0~%# Target is too close to the sensor~%int8 ERROR_NEAR = 1~%# Target is too far from the sensor (> 2047mm)~%int8 ERROR_FAR = 2~%# Other general error~%int8 ERROR_OTHER = 3~%~%int8 error       # One of NO_ERROR, ERROR_NEAR, ERROR_FAR, or ERROR_OTHER~%int16 distance   # Distance in mm. Only valid if error == NO_ERROR~%int16 confidence # An abstract \"confidence\" measurement that is not well-defined in the RFD77402 datasheet~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ToFStamped)))
  "Returns full string definition for message of type 'ToFStamped"
  (cl:format cl:nil "Header header~%~%# There is no error. `distance` is valid.~%int8 NO_ERROR = 0~%# Target is too close to the sensor~%int8 ERROR_NEAR = 1~%# Target is too far from the sensor (> 2047mm)~%int8 ERROR_FAR = 2~%# Other general error~%int8 ERROR_OTHER = 3~%~%int8 error       # One of NO_ERROR, ERROR_NEAR, ERROR_FAR, or ERROR_OTHER~%int16 distance   # Distance in mm. Only valid if error == NO_ERROR~%int16 confidence # An abstract \"confidence\" measurement that is not well-defined in the RFD77402 datasheet~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ToFStamped>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     2
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ToFStamped>))
  "Converts a ROS message object to a list"
  (cl:list 'ToFStamped
    (cl:cons ':header (header msg))
    (cl:cons ':error (error msg))
    (cl:cons ':distance (distance msg))
    (cl:cons ':confidence (confidence msg))
))
