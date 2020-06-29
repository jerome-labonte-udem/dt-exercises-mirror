;; Auto-generated. Do not edit!


(when (boundp 'ground_projection::EstimateHomography)
  (if (not (find-package "GROUND_PROJECTION"))
    (make-package "GROUND_PROJECTION"))
  (shadow 'EstimateHomography (find-package "GROUND_PROJECTION")))
(unless (find-package "GROUND_PROJECTION::ESTIMATEHOMOGRAPHY")
  (make-package "GROUND_PROJECTION::ESTIMATEHOMOGRAPHY"))
(unless (find-package "GROUND_PROJECTION::ESTIMATEHOMOGRAPHYREQUEST")
  (make-package "GROUND_PROJECTION::ESTIMATEHOMOGRAPHYREQUEST"))
(unless (find-package "GROUND_PROJECTION::ESTIMATEHOMOGRAPHYRESPONSE")
  (make-package "GROUND_PROJECTION::ESTIMATEHOMOGRAPHYRESPONSE"))

(in-package "ROS")





(defclass ground_projection::EstimateHomographyRequest
  :super ros::object
  :slots ())

(defmethod ground_projection::EstimateHomographyRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass ground_projection::EstimateHomographyResponse
  :super ros::object
  :slots ())

(defmethod ground_projection::EstimateHomographyResponse
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass ground_projection::EstimateHomography
  :super ros::object
  :slots ())

(setf (get ground_projection::EstimateHomography :md5sum-) "d41d8cd98f00b204e9800998ecf8427e")
(setf (get ground_projection::EstimateHomography :datatype-) "ground_projection/EstimateHomography")
(setf (get ground_projection::EstimateHomography :request) ground_projection::EstimateHomographyRequest)
(setf (get ground_projection::EstimateHomography :response) ground_projection::EstimateHomographyResponse)

(defmethod ground_projection::EstimateHomographyRequest
  (:response () (instance ground_projection::EstimateHomographyResponse :init)))

(setf (get ground_projection::EstimateHomographyRequest :md5sum-) "d41d8cd98f00b204e9800998ecf8427e")
(setf (get ground_projection::EstimateHomographyRequest :datatype-) "ground_projection/EstimateHomographyRequest")
(setf (get ground_projection::EstimateHomographyRequest :definition-)
      "
---

")

(setf (get ground_projection::EstimateHomographyResponse :md5sum-) "d41d8cd98f00b204e9800998ecf8427e")
(setf (get ground_projection::EstimateHomographyResponse :datatype-) "ground_projection/EstimateHomographyResponse")
(setf (get ground_projection::EstimateHomographyResponse :definition-)
      "
---

")



(provide :ground_projection/EstimateHomography "d41d8cd98f00b204e9800998ecf8427e")


