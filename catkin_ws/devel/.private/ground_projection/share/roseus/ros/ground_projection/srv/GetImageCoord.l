;; Auto-generated. Do not edit!


(when (boundp 'ground_projection::GetImageCoord)
  (if (not (find-package "GROUND_PROJECTION"))
    (make-package "GROUND_PROJECTION"))
  (shadow 'GetImageCoord (find-package "GROUND_PROJECTION")))
(unless (find-package "GROUND_PROJECTION::GETIMAGECOORD")
  (make-package "GROUND_PROJECTION::GETIMAGECOORD"))
(unless (find-package "GROUND_PROJECTION::GETIMAGECOORDREQUEST")
  (make-package "GROUND_PROJECTION::GETIMAGECOORDREQUEST"))
(unless (find-package "GROUND_PROJECTION::GETIMAGECOORDRESPONSE")
  (make-package "GROUND_PROJECTION::GETIMAGECOORDRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "DUCKIETOWN_MSGS"))
  (ros::roseus-add-msgs "duckietown_msgs"))


(defclass ground_projection::GetImageCoordRequest
  :super ros::object
  :slots (_gp ))

(defmethod ground_projection::GetImageCoordRequest
  (:init
   (&key
    ((:gp __gp) (instance geometry_msgs::Point :init))
    )
   (send-super :init)
   (setq _gp __gp)
   self)
  (:gp
   (&rest __gp)
   (if (keywordp (car __gp))
       (send* _gp __gp)
     (progn
       (if __gp (setq _gp (car __gp)))
       _gp)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Point _gp
    (send _gp :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Point _gp
       (send _gp :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Point _gp
     (send _gp :deserialize buf ptr-) (incf ptr- (send _gp :serialization-length))
   ;;
   self)
  )

(defclass ground_projection::GetImageCoordResponse
  :super ros::object
  :slots (_normalized_uv ))

(defmethod ground_projection::GetImageCoordResponse
  (:init
   (&key
    ((:normalized_uv __normalized_uv) (instance duckietown_msgs::Vector2D :init))
    )
   (send-super :init)
   (setq _normalized_uv __normalized_uv)
   self)
  (:normalized_uv
   (&rest __normalized_uv)
   (if (keywordp (car __normalized_uv))
       (send* _normalized_uv __normalized_uv)
     (progn
       (if __normalized_uv (setq _normalized_uv (car __normalized_uv)))
       _normalized_uv)))
  (:serialization-length
   ()
   (+
    ;; duckietown_msgs/Vector2D _normalized_uv
    (send _normalized_uv :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; duckietown_msgs/Vector2D _normalized_uv
       (send _normalized_uv :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; duckietown_msgs/Vector2D _normalized_uv
     (send _normalized_uv :deserialize buf ptr-) (incf ptr- (send _normalized_uv :serialization-length))
   ;;
   self)
  )

(defclass ground_projection::GetImageCoord
  :super ros::object
  :slots ())

(setf (get ground_projection::GetImageCoord :md5sum-) "590f09a60b9d5d3e1b4384278b6e2b2f")
(setf (get ground_projection::GetImageCoord :datatype-) "ground_projection/GetImageCoord")
(setf (get ground_projection::GetImageCoord :request) ground_projection::GetImageCoordRequest)
(setf (get ground_projection::GetImageCoord :response) ground_projection::GetImageCoordResponse)

(defmethod ground_projection::GetImageCoordRequest
  (:response () (instance ground_projection::GetImageCoordResponse :init)))

(setf (get ground_projection::GetImageCoordRequest :md5sum-) "590f09a60b9d5d3e1b4384278b6e2b2f")
(setf (get ground_projection::GetImageCoordRequest :datatype-) "ground_projection/GetImageCoordRequest")
(setf (get ground_projection::GetImageCoordRequest :definition-)
      "
geometry_msgs/Point gp

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---


duckietown_msgs/Vector2D normalized_uv

================================================================================
MSG: duckietown_msgs/Vector2D
float32 x
float32 y
")

(setf (get ground_projection::GetImageCoordResponse :md5sum-) "590f09a60b9d5d3e1b4384278b6e2b2f")
(setf (get ground_projection::GetImageCoordResponse :datatype-) "ground_projection/GetImageCoordResponse")
(setf (get ground_projection::GetImageCoordResponse :definition-)
      "
geometry_msgs/Point gp

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---


duckietown_msgs/Vector2D normalized_uv

================================================================================
MSG: duckietown_msgs/Vector2D
float32 x
float32 y
")



(provide :ground_projection/GetImageCoord "590f09a60b9d5d3e1b4384278b6e2b2f")


