;; Auto-generated. Do not edit!


(when (boundp 'ground_projection::GetGroundCoord)
  (if (not (find-package "GROUND_PROJECTION"))
    (make-package "GROUND_PROJECTION"))
  (shadow 'GetGroundCoord (find-package "GROUND_PROJECTION")))
(unless (find-package "GROUND_PROJECTION::GETGROUNDCOORD")
  (make-package "GROUND_PROJECTION::GETGROUNDCOORD"))
(unless (find-package "GROUND_PROJECTION::GETGROUNDCOORDREQUEST")
  (make-package "GROUND_PROJECTION::GETGROUNDCOORDREQUEST"))
(unless (find-package "GROUND_PROJECTION::GETGROUNDCOORDRESPONSE")
  (make-package "GROUND_PROJECTION::GETGROUNDCOORDRESPONSE"))

(in-package "ROS")

(if (not (find-package "DUCKIETOWN_MSGS"))
  (ros::roseus-add-msgs "duckietown_msgs"))


(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass ground_projection::GetGroundCoordRequest
  :super ros::object
  :slots (_uv ))

(defmethod ground_projection::GetGroundCoordRequest
  (:init
   (&key
    ((:uv __uv) (instance duckietown_msgs::Pixel :init))
    )
   (send-super :init)
   (setq _uv __uv)
   self)
  (:uv
   (&rest __uv)
   (if (keywordp (car __uv))
       (send* _uv __uv)
     (progn
       (if __uv (setq _uv (car __uv)))
       _uv)))
  (:serialization-length
   ()
   (+
    ;; duckietown_msgs/Pixel _uv
    (send _uv :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; duckietown_msgs/Pixel _uv
       (send _uv :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; duckietown_msgs/Pixel _uv
     (send _uv :deserialize buf ptr-) (incf ptr- (send _uv :serialization-length))
   ;;
   self)
  )

(defclass ground_projection::GetGroundCoordResponse
  :super ros::object
  :slots (_gp ))

(defmethod ground_projection::GetGroundCoordResponse
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

(defclass ground_projection::GetGroundCoord
  :super ros::object
  :slots ())

(setf (get ground_projection::GetGroundCoord :md5sum-) "4a593ebeb349392dda67c9ac9b35b490")
(setf (get ground_projection::GetGroundCoord :datatype-) "ground_projection/GetGroundCoord")
(setf (get ground_projection::GetGroundCoord :request) ground_projection::GetGroundCoordRequest)
(setf (get ground_projection::GetGroundCoord :response) ground_projection::GetGroundCoordResponse)

(defmethod ground_projection::GetGroundCoordRequest
  (:response () (instance ground_projection::GetGroundCoordResponse :init)))

(setf (get ground_projection::GetGroundCoordRequest :md5sum-) "4a593ebeb349392dda67c9ac9b35b490")
(setf (get ground_projection::GetGroundCoordRequest :datatype-) "ground_projection/GetGroundCoordRequest")
(setf (get ground_projection::GetGroundCoordRequest :definition-)
      "
duckietown_msgs/Pixel uv


================================================================================
MSG: duckietown_msgs/Pixel
int32 u
---

geometry_msgs/Point gp


================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
")

(setf (get ground_projection::GetGroundCoordResponse :md5sum-) "4a593ebeb349392dda67c9ac9b35b490")
(setf (get ground_projection::GetGroundCoordResponse :datatype-) "ground_projection/GetGroundCoordResponse")
(setf (get ground_projection::GetGroundCoordResponse :definition-)
      "
duckietown_msgs/Pixel uv


================================================================================
MSG: duckietown_msgs/Pixel
int32 u
---

geometry_msgs/Point gp


================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
")



(provide :ground_projection/GetGroundCoord "4a593ebeb349392dda67c9ac9b35b490")


