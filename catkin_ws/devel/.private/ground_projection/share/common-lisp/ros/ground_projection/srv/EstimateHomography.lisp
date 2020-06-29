; Auto-generated. Do not edit!


(cl:in-package ground_projection-srv)


;//! \htmlinclude EstimateHomography-request.msg.html

(cl:defclass <EstimateHomography-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass EstimateHomography-request (<EstimateHomography-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EstimateHomography-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EstimateHomography-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ground_projection-srv:<EstimateHomography-request> is deprecated: use ground_projection-srv:EstimateHomography-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EstimateHomography-request>) ostream)
  "Serializes a message object of type '<EstimateHomography-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EstimateHomography-request>) istream)
  "Deserializes a message object of type '<EstimateHomography-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EstimateHomography-request>)))
  "Returns string type for a service object of type '<EstimateHomography-request>"
  "ground_projection/EstimateHomographyRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EstimateHomography-request)))
  "Returns string type for a service object of type 'EstimateHomography-request"
  "ground_projection/EstimateHomographyRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EstimateHomography-request>)))
  "Returns md5sum for a message object of type '<EstimateHomography-request>"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EstimateHomography-request)))
  "Returns md5sum for a message object of type 'EstimateHomography-request"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EstimateHomography-request>)))
  "Returns full string definition for message of type '<EstimateHomography-request>"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EstimateHomography-request)))
  "Returns full string definition for message of type 'EstimateHomography-request"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EstimateHomography-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EstimateHomography-request>))
  "Converts a ROS message object to a list"
  (cl:list 'EstimateHomography-request
))
;//! \htmlinclude EstimateHomography-response.msg.html

(cl:defclass <EstimateHomography-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass EstimateHomography-response (<EstimateHomography-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EstimateHomography-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EstimateHomography-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ground_projection-srv:<EstimateHomography-response> is deprecated: use ground_projection-srv:EstimateHomography-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EstimateHomography-response>) ostream)
  "Serializes a message object of type '<EstimateHomography-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EstimateHomography-response>) istream)
  "Deserializes a message object of type '<EstimateHomography-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EstimateHomography-response>)))
  "Returns string type for a service object of type '<EstimateHomography-response>"
  "ground_projection/EstimateHomographyResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EstimateHomography-response)))
  "Returns string type for a service object of type 'EstimateHomography-response"
  "ground_projection/EstimateHomographyResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EstimateHomography-response>)))
  "Returns md5sum for a message object of type '<EstimateHomography-response>"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EstimateHomography-response)))
  "Returns md5sum for a message object of type 'EstimateHomography-response"
  "d41d8cd98f00b204e9800998ecf8427e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EstimateHomography-response>)))
  "Returns full string definition for message of type '<EstimateHomography-response>"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EstimateHomography-response)))
  "Returns full string definition for message of type 'EstimateHomography-response"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EstimateHomography-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EstimateHomography-response>))
  "Converts a ROS message object to a list"
  (cl:list 'EstimateHomography-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'EstimateHomography)))
  'EstimateHomography-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'EstimateHomography)))
  'EstimateHomography-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EstimateHomography)))
  "Returns string type for a service object of type '<EstimateHomography>"
  "ground_projection/EstimateHomography")