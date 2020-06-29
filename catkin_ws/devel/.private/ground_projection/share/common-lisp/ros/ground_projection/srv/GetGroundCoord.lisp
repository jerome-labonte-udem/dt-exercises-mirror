; Auto-generated. Do not edit!


(cl:in-package ground_projection-srv)


;//! \htmlinclude GetGroundCoord-request.msg.html

(cl:defclass <GetGroundCoord-request> (roslisp-msg-protocol:ros-message)
  ((uv
    :reader uv
    :initarg :uv
    :type duckietown_msgs-msg:Pixel
    :initform (cl:make-instance 'duckietown_msgs-msg:Pixel)))
)

(cl:defclass GetGroundCoord-request (<GetGroundCoord-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetGroundCoord-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetGroundCoord-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ground_projection-srv:<GetGroundCoord-request> is deprecated: use ground_projection-srv:GetGroundCoord-request instead.")))

(cl:ensure-generic-function 'uv-val :lambda-list '(m))
(cl:defmethod uv-val ((m <GetGroundCoord-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ground_projection-srv:uv-val is deprecated.  Use ground_projection-srv:uv instead.")
  (uv m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetGroundCoord-request>) ostream)
  "Serializes a message object of type '<GetGroundCoord-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'uv) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetGroundCoord-request>) istream)
  "Deserializes a message object of type '<GetGroundCoord-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'uv) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetGroundCoord-request>)))
  "Returns string type for a service object of type '<GetGroundCoord-request>"
  "ground_projection/GetGroundCoordRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetGroundCoord-request)))
  "Returns string type for a service object of type 'GetGroundCoord-request"
  "ground_projection/GetGroundCoordRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetGroundCoord-request>)))
  "Returns md5sum for a message object of type '<GetGroundCoord-request>"
  "4a593ebeb349392dda67c9ac9b35b490")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetGroundCoord-request)))
  "Returns md5sum for a message object of type 'GetGroundCoord-request"
  "4a593ebeb349392dda67c9ac9b35b490")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetGroundCoord-request>)))
  "Returns full string definition for message of type '<GetGroundCoord-request>"
  (cl:format cl:nil "~%duckietown_msgs/Pixel uv~%~%~%================================================================================~%MSG: duckietown_msgs/Pixel~%int32 u~%int32 v~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetGroundCoord-request)))
  "Returns full string definition for message of type 'GetGroundCoord-request"
  (cl:format cl:nil "~%duckietown_msgs/Pixel uv~%~%~%================================================================================~%MSG: duckietown_msgs/Pixel~%int32 u~%int32 v~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetGroundCoord-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'uv))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetGroundCoord-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetGroundCoord-request
    (cl:cons ':uv (uv msg))
))
;//! \htmlinclude GetGroundCoord-response.msg.html

(cl:defclass <GetGroundCoord-response> (roslisp-msg-protocol:ros-message)
  ((gp
    :reader gp
    :initarg :gp
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point)))
)

(cl:defclass GetGroundCoord-response (<GetGroundCoord-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetGroundCoord-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetGroundCoord-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ground_projection-srv:<GetGroundCoord-response> is deprecated: use ground_projection-srv:GetGroundCoord-response instead.")))

(cl:ensure-generic-function 'gp-val :lambda-list '(m))
(cl:defmethod gp-val ((m <GetGroundCoord-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ground_projection-srv:gp-val is deprecated.  Use ground_projection-srv:gp instead.")
  (gp m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetGroundCoord-response>) ostream)
  "Serializes a message object of type '<GetGroundCoord-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'gp) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetGroundCoord-response>) istream)
  "Deserializes a message object of type '<GetGroundCoord-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'gp) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetGroundCoord-response>)))
  "Returns string type for a service object of type '<GetGroundCoord-response>"
  "ground_projection/GetGroundCoordResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetGroundCoord-response)))
  "Returns string type for a service object of type 'GetGroundCoord-response"
  "ground_projection/GetGroundCoordResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetGroundCoord-response>)))
  "Returns md5sum for a message object of type '<GetGroundCoord-response>"
  "4a593ebeb349392dda67c9ac9b35b490")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetGroundCoord-response)))
  "Returns md5sum for a message object of type 'GetGroundCoord-response"
  "4a593ebeb349392dda67c9ac9b35b490")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetGroundCoord-response>)))
  "Returns full string definition for message of type '<GetGroundCoord-response>"
  (cl:format cl:nil "~%geometry_msgs/Point gp~%~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetGroundCoord-response)))
  "Returns full string definition for message of type 'GetGroundCoord-response"
  (cl:format cl:nil "~%geometry_msgs/Point gp~%~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetGroundCoord-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'gp))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetGroundCoord-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetGroundCoord-response
    (cl:cons ':gp (gp msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetGroundCoord)))
  'GetGroundCoord-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetGroundCoord)))
  'GetGroundCoord-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetGroundCoord)))
  "Returns string type for a service object of type '<GetGroundCoord>"
  "ground_projection/GetGroundCoord")