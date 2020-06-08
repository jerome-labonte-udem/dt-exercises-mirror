; Auto-generated. Do not edit!


(cl:in-package ground_projection-srv)


;//! \htmlinclude GetImageCoord-request.msg.html

(cl:defclass <GetImageCoord-request> (roslisp-msg-protocol:ros-message)
  ((gp
    :reader gp
    :initarg :gp
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point)))
)

(cl:defclass GetImageCoord-request (<GetImageCoord-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetImageCoord-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetImageCoord-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ground_projection-srv:<GetImageCoord-request> is deprecated: use ground_projection-srv:GetImageCoord-request instead.")))

(cl:ensure-generic-function 'gp-val :lambda-list '(m))
(cl:defmethod gp-val ((m <GetImageCoord-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ground_projection-srv:gp-val is deprecated.  Use ground_projection-srv:gp instead.")
  (gp m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetImageCoord-request>) ostream)
  "Serializes a message object of type '<GetImageCoord-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'gp) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetImageCoord-request>) istream)
  "Deserializes a message object of type '<GetImageCoord-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'gp) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetImageCoord-request>)))
  "Returns string type for a service object of type '<GetImageCoord-request>"
  "ground_projection/GetImageCoordRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetImageCoord-request)))
  "Returns string type for a service object of type 'GetImageCoord-request"
  "ground_projection/GetImageCoordRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetImageCoord-request>)))
  "Returns md5sum for a message object of type '<GetImageCoord-request>"
  "590f09a60b9d5d3e1b4384278b6e2b2f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetImageCoord-request)))
  "Returns md5sum for a message object of type 'GetImageCoord-request"
  "590f09a60b9d5d3e1b4384278b6e2b2f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetImageCoord-request>)))
  "Returns full string definition for message of type '<GetImageCoord-request>"
  (cl:format cl:nil "~%geometry_msgs/Point gp~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetImageCoord-request)))
  "Returns full string definition for message of type 'GetImageCoord-request"
  (cl:format cl:nil "~%geometry_msgs/Point gp~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetImageCoord-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'gp))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetImageCoord-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetImageCoord-request
    (cl:cons ':gp (gp msg))
))
;//! \htmlinclude GetImageCoord-response.msg.html

(cl:defclass <GetImageCoord-response> (roslisp-msg-protocol:ros-message)
  ((normalized_uv
    :reader normalized_uv
    :initarg :normalized_uv
    :type duckietown_msgs-msg:Vector2D
    :initform (cl:make-instance 'duckietown_msgs-msg:Vector2D)))
)

(cl:defclass GetImageCoord-response (<GetImageCoord-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetImageCoord-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetImageCoord-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ground_projection-srv:<GetImageCoord-response> is deprecated: use ground_projection-srv:GetImageCoord-response instead.")))

(cl:ensure-generic-function 'normalized_uv-val :lambda-list '(m))
(cl:defmethod normalized_uv-val ((m <GetImageCoord-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ground_projection-srv:normalized_uv-val is deprecated.  Use ground_projection-srv:normalized_uv instead.")
  (normalized_uv m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetImageCoord-response>) ostream)
  "Serializes a message object of type '<GetImageCoord-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'normalized_uv) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetImageCoord-response>) istream)
  "Deserializes a message object of type '<GetImageCoord-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'normalized_uv) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetImageCoord-response>)))
  "Returns string type for a service object of type '<GetImageCoord-response>"
  "ground_projection/GetImageCoordResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetImageCoord-response)))
  "Returns string type for a service object of type 'GetImageCoord-response"
  "ground_projection/GetImageCoordResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetImageCoord-response>)))
  "Returns md5sum for a message object of type '<GetImageCoord-response>"
  "590f09a60b9d5d3e1b4384278b6e2b2f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetImageCoord-response)))
  "Returns md5sum for a message object of type 'GetImageCoord-response"
  "590f09a60b9d5d3e1b4384278b6e2b2f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetImageCoord-response>)))
  "Returns full string definition for message of type '<GetImageCoord-response>"
  (cl:format cl:nil "~%~%duckietown_msgs/Vector2D normalized_uv~%~%================================================================================~%MSG: duckietown_msgs/Vector2D~%float32 x~%float32 y~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetImageCoord-response)))
  "Returns full string definition for message of type 'GetImageCoord-response"
  (cl:format cl:nil "~%~%duckietown_msgs/Vector2D normalized_uv~%~%================================================================================~%MSG: duckietown_msgs/Vector2D~%float32 x~%float32 y~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetImageCoord-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'normalized_uv))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetImageCoord-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetImageCoord-response
    (cl:cons ':normalized_uv (normalized_uv msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetImageCoord)))
  'GetImageCoord-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetImageCoord)))
  'GetImageCoord-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetImageCoord)))
  "Returns string type for a service object of type '<GetImageCoord>"
  "ground_projection/GetImageCoord")