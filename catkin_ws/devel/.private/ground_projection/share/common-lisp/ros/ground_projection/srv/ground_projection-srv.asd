
(cl:in-package :asdf)

(defsystem "ground_projection-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :duckietown_msgs-msg
               :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "EstimateHomography" :depends-on ("_package_EstimateHomography"))
    (:file "_package_EstimateHomography" :depends-on ("_package"))
    (:file "GetGroundCoord" :depends-on ("_package_GetGroundCoord"))
    (:file "_package_GetGroundCoord" :depends-on ("_package"))
    (:file "GetImageCoord" :depends-on ("_package_GetImageCoord"))
    (:file "_package_GetImageCoord" :depends-on ("_package"))
  ))