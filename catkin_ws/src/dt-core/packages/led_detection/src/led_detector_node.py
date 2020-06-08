#!/usr/bin/env python
import rospy
import time
from led_detection.LEDDetector import LEDDetector
from std_msgs.msg import Byte
from duckietown_msgs.msg import Vector2D, LEDDetection, LEDDetectionArray, LEDDetectionDebugInfo, BoolStamped, SignalsDetection
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from duckietown_utils.bag_logs import numpy_from_ros_compressed
import numpy as np
import cv2
import scipy.fftpack
from cv_bridge import CvBridge, CvBridgeError

class LEDDetectorNode(object):
    def __init__(self):
        self.active = True # [INTERACTIVE MODE] Won't be overwritten if FSM isn't running, node always active
        self.first_timestamp = 0
        self.capture_finished = True
        self.tinit = None
        self.trigger = True
        self.node_state = 0
        self.data = []

        # Needed to publish images
        self.bridge = CvBridge()

        # Node name
        self.node_name = rospy.get_name()

        # Capture time
        self.capture_time = 0.5

        # Parameters
        self.DTOL = 15

        # Use FFT or heuristics
        self.useFFT = True
        self.freqIdentify = []

        # Cropping
        self.cropNormalizedRight = [[0.1,0.67],[0.6,1.0]]
        self.cropNormalizedFront = [[0.1,0.5],[0.13,0.5]]
        self.cropNormalizedTL    = [[0.0,0.25],[0.25,0.75]]

        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()  # Change thresholds
        params.minThreshold = 5
        # params.maxThreshold = 200
        params.maxThreshold = 75
        params.thresholdStep = 10

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10*10*3.14
        params.maxArea = 20*20*3.14

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.05

        # Parameters
        params_car = params
        params_tl  = params

        # Change parameters for the traffic light
        params_tl.minArea = 5*5*3.14
        params_tl.maxArea = 15*15*3.14

        # Create a detector with the parameters
        self.detector_car = cv2.SimpleBlobDetector_create(params_car)
        self.detector_tl  = cv2.SimpleBlobDetector_create(params_tl)

        # Publish
        self.pub_raw_detections = rospy.Publisher("~raw_led_detection",LEDDetectionArray,queue_size=1)
        self.pub_image_right    = rospy.Publisher("~image_detection_right",Image,queue_size=1)
        self.pub_image_front    = rospy.Publisher("~image_detection_front", Image, queue_size=1)
        self.pub_image_TL       = rospy.Publisher("~image_detection_TL", Image, queue_size=1)
        self.pub_detections     = rospy.Publisher("~led_detection", SignalsDetection, queue_size=1)
        self.pub_debug          = rospy.Publisher("~debug_info",LEDDetectionDebugInfo,queue_size=1)
        self.veh_name           = rospy.get_namespace().strip("/")

        # Subscribed
        self.sub_cam    = rospy.Subscriber("camera_node/image/compressed", CompressedImage, self.camera_callback)
        self.sub_trig   = rospy.Subscriber("~trigger",Byte, self.trigger_callback)
        self.sub_switch = rospy.Subscriber("~switch",BoolStamped,self.cbSwitch)

        # Additional parameters
        self.protocol            = rospy.get_param("~LED_protocol")
        #self.capture_time         = rospy.get_param("~capture_time")
        self.continuous           = rospy.get_param('~continuous', False) # Detect continuously as long as active
                                                               # [INTERACTIVE MODE] set to False for manual trigger
        # Cell size (needed for visualization)
        self.cell_size      = rospy.get_param("~cell_size")
        self.crop_rect_norm = rospy.get_param("~crop_rect_normalized")

        # Get frequency to indentify
        self.freqIdentify = [self.protocol['signals']['CAR_SIGNAL_A']['frequency'],
                             self.protocol['signals']['CAR_SIGNAL_PRIORITY']['frequency'],
                             self.protocol['signals']['CAR_SIGNAL_SACRIFICE_FOR_PRIORITY']['frequency']]
        #print '---------------------------------------------------------------'
        #print self.freqIdentify
        #print '---------------------------------------------------------------'

        #rospy.loginfo('[%s] Config: \n\t crop_rect_normalized: %s, \n\t capture_time: %s, \n\t cell_size: %s'%(self.node_name, self.crop_rect_normalized, self.capture_time, self.cell_size))

        # Check vehicle name
        if not self.veh_name:
            # fall back on private param passed thru rosrun
            # syntax is: rosrun <pkg> <node> _veh:=<bot-id>
            if rospy.has_param('~veh'):
                self.veh_name = rospy.get_param('~veh')

        if not self.veh_name:
            raise ValueError('Vehicle name is not set.')

        # Loginfo
        rospy.loginfo('[%s] Vehicle: %s'%(self.node_name, self.veh_name))
        rospy.loginfo('[%s] Waiting for camera image...' %self.node_name)

    def cbSwitch(self, switch_msg): # active/inactive switch from FSM
        self.active = switch_msg.data
        if self.active:
            self.trigger = True

    def camera_callback(self, msg):
        if not self.active:
            return

        float_time = msg.header.stamp.to_sec()
        debug_msg  = LEDDetectionDebugInfo()

        if self.trigger:
            #rospy.loginfo('[%s] GOT TRIGGER! Starting...')
            self.trigger          = False
            self.data             = []
            self.capture_finished = False
            # Start capturing images
            #rospy.loginfo('[%s] Start capturing frames'%self.node_name)
            self.first_timestamp = msg.header.stamp.to_sec()
            self.tinit           = time.time()

        elif self.capture_finished:
            self.node_state = 0
            #rospy.loginfo('[%s] Waiting for trigger...' %self.node_name)

        if self.first_timestamp > 0:
            # TODO sanity check rel_time positive, restart otherwise
            rel_time = float_time - self.first_timestamp

            # Capturing
            if rel_time < self.capture_time:
                self.node_state = 1
                # Capture image
                rgb = numpy_from_ros_compressed(msg)
                rgb = cv2.cvtColor(rgb,cv2.COLOR_BGRA2GRAY)
                rgb = cv2.resize(rgb, (640 * 1, 480 * 1))
                rgb = 255 - rgb
                #rospy.loginfo('[%s] Capturing frame %s' %(self.node_name, rel_time))
                # Save image to data
                #if np.size(self.data) == 0:
                #    self.data = rgb
                #else:
                #    self.data = np.dstack((self.data,rgb))
                self.data.append({'timestamp': float_time, 'rgb': rgb[:,:]})
                debug_msg.capture_progress = 100.0*rel_time/self.capture_time

            # Start processing
            elif not self.capture_finished and self.first_timestamp > 0:
                #rospy.loginfo('[%s] Relative Time %s, processing' %(self.node_name, rel_time))
                self.node_state = 2
                self.capture_finished = True
                self.first_timestamp = 0
                self.sub_cam.unregister() # IMPORTANT! Explicitly ignore messages
                                          # while processing, accumulates delay otherwise!
                self.send_state(debug_msg)
                # Process image and publish results
                self.process_and_publish()

        self.send_state(debug_msg) # TODO move heartbeat to dedicated thread

    def trigger_callback(self, msg):
        self.trigger = True


    def crop_image(self,images,cropNorm):
        # Get size
        H,W,_ = images.shape
        # Compute indices
        hStart = int(np.floor(H*cropNorm[0][0]))
        hEnd   = int(np.ceil(H*cropNorm[0][1]))
        wStart = int(np.floor(W*cropNorm[1][0]))
        wEnd   = int(np.ceil(W*cropNorm[1][1]))
        # Crop image
        imageCropped = images[hStart:hEnd,wStart:wEnd,:]
        # Return cropped image
        return imageCropped

    def process_and_publish(self):
        # Initial time
        tic = time.time()

        # Get dimensions
        H,W = self.data[0]['rgb'].shape
        NIm = len(self.data)

        # Save in proper vectors
        images     = np.zeros((H,W,NIm),dtype=np.uint8)
        timestamps = np.zeros((NIm))
        for i, v in enumerate(self.data):
            timestamps[i] = v['timestamp']
            images[:,:,i] = v['rgb']

        # Crop images
        imRight = self.crop_image(images,self.cropNormalizedRight)
        imFront = self.crop_image(images,self.cropNormalizedFront)
        imTL    = self.crop_image(images,self.cropNormalizedTL)

        # Allocate space
        FrameRight = []
        BlobsRight = []
        FrameFront = []
        BlobsFront = []
        FrameTL    = []
        BlobsTL    = []

        # Print on screen
        #rospy.loginfo('[%s] Analyzing %s images of size %s X %s' %(self.node_name,NIm,W,H))

        # Iterate
        for t in range(NIm):
            # Iterate Right
            # Detect blobs.
            keypoints = self.detector_car.detect(imRight[:, :, t])
            FrameRight.append(np.zeros((2, len(keypoints))))

            for n in range(len(keypoints)):
                FrameRight[t][:, n] = keypoints[n].pt
                if len(BlobsRight) == 0:
                    # If no blobs saved, then save the first LED detected
                    BlobsRight.append({'p': FrameRight[t][:, n], 'N': 1, 'Signal': np.zeros(imRight.shape[2])})
                    BlobsRight[-1]['Signal'][t] = 1
                else:
                    # Thereafter, check whether the detected LED belongs to a blob
                    Distance = np.empty(len(BlobsRight))
                    for k in range(len(BlobsRight)):
                        Distance[k] = np.linalg.norm(BlobsRight[k]['p'] - FrameRight[t][:, n])
                    if np.min(Distance) < self.DTOL:
                        if BlobsRight[np.argmin(Distance)]['Signal'][t] == 0:
                            BlobsRight[np.argmin(Distance)]['N'] += 1
                            BlobsRight[np.argmin(Distance)]['Signal'][t] = 1
                    else:
                        BlobsRight.append({'p': FrameRight[t][:, n], 'N': 1, 'Signal': np.zeros(imRight.shape[2])})
                        BlobsRight[-1]['Signal'][t] = 1

            # Iterate Front
            # Detect blobs.
            keypoints = self.detector_car.detect(imFront[:, :, t])
            FrameFront.append(np.zeros((2, len(keypoints))))

            for n in range(len(keypoints)):
                FrameFront[t][:, n] = keypoints[n].pt
                if len(BlobsFront) == 0:
                    # If no blobs saved, then save the first LED detected
                    BlobsFront.append({'p': FrameFront[t][:, n], 'N': 1, 'Signal': np.zeros(imFront.shape[2])})
                    BlobsFront[-1]['Signal'][t] = 1
                else:
                    # Thereafter, check whether the detected LED belongs to a blob
                    Distance = np.empty(len(BlobsFront))
                    for k in range(len(BlobsFront)):
                        Distance[k] = np.linalg.norm(BlobsFront[k]['p'] - FrameFront[t][:, n])
                    if np.min(Distance) < self.DTOL:
                        if BlobsFront[np.argmin(Distance)]['Signal'][t] == 0:
                            BlobsFront[np.argmin(Distance)]['N'] += 1
                            BlobsFront[np.argmin(Distance)]['Signal'][t] = 1
                    else:
                        BlobsFront.append({'p': FrameFront[t][:, n], 'N': 1, 'Signal': np.zeros(imFront.shape[2])})
                        BlobsFront[-1]['Signal'][t] = 1

            # Iterate TL
            # Detect blobs.
            keypoints = self.detector_tl.detect(imTL[:, :, t])
            FrameTL.append(np.zeros((2, len(keypoints))))

            for n in range(len(keypoints)):
                FrameTL[t][:, n] = keypoints[n].pt
                if len(BlobsTL) == 0:
                    # If no blobs saved, then save the first LED detected
                    BlobsTL.append(
                        {'p': FrameTL[t][:, n], 'N': 1, 'Signal': np.zeros(imTL.shape[2])})
                    BlobsTL[-1]['Signal'][t] = 1
                else:
                    # Thereafter, check whether the detected LED belongs to a blob
                    Distance = np.empty(len(BlobsTL))
                    for k in range(len(BlobsTL)):
                        Distance[k] = np.linalg.norm(BlobsTL[k]['p'] - FrameTL[t][:, n])
                    if np.min(Distance) < self.DTOL:
                        if BlobsTL[np.argmin(Distance)]['Signal'][t] == 0:
                            BlobsTL[np.argmin(Distance)]['N'] += 1
                            BlobsTL[np.argmin(Distance)]['Signal'][t] = 1
                    else:
                        BlobsTL.append(
                            {'p': FrameTL[t][:, n], 'N': 1, 'Signal': np.zeros(imTL.shape[2])})
                        BlobsTL[-1]['Signal'][t] = 1

        # Extract blobs (right)
        keypointBlobRight = []
        radiusRight       = self.DTOL/2.0
        for k in range(len(BlobsRight)):
            assert np.sum(BlobsRight[k]['Signal']) == BlobsRight[k]['N']
            keypointBlobRight.append(cv2.KeyPoint(BlobsRight[k]['p'][0], BlobsRight[k]['p'][1], radiusRight))

        # Extract blobs (front)
        keypointBlobFront = []
        radiusFront       = self.DTOL/2.0
        for k in range(len(BlobsFront)):
            assert np.sum(BlobsFront[k]['Signal']) == BlobsFront[k]['N']
            keypointBlobFront.append(cv2.KeyPoint(BlobsFront[k]['p'][0], BlobsFront[k]['p'][1], radiusFront))

        # Extract blobs (TL)
        keypointBlobTL = []
        radiusTL       = self.DTOL/2.0
        for k in range(len(BlobsTL)):
            assert np.sum(BlobsTL[k]['Signal']) == BlobsTL[k]['N']
            keypointBlobTL.append(cv2.KeyPoint(BlobsTL[k]['p'][0], BlobsTL[k]['p'][1], radiusTL))

        # Images
        imPublishRight = cv2.drawKeypoints(imRight[:,:,-1], keypointBlobRight, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imPublishFront = cv2.drawKeypoints(imFront[:,:,-1], keypointBlobFront, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imPublishTL    = cv2.drawKeypoints(imTL[:,:,-1], keypointBlobTL, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Initialize detection
        self.right = SignalsDetection.NO_CAR
        self.front = SignalsDetection.NO_CAR
        self.traffic_light = SignalsDetection.NO_TRAFFIC_LIGHT

        # Result
        result = LEDDetectionArray()

        # Sampling time
        T = (1.0*self.capture_time)/(1.0*NIm)

        # Decide whether LED or not (right)
        for i in range(len(BlobsRight)):
            #rospy.loginfo('[%s] Detection on the right' % (self.node_name))
            # Detection
            detected,result,freq_identified, fft_peak_freq = self.detect_blob(BlobsRight[i],T,NIm,H,W,self.cropNormalizedRight,timestamps,result)

            # Take decision
            if detected:

                #print '-------------------'
                #print("NIm = %d " % NIm)
                #print("T = %f " % T)
                #print("fft_peak_freq = %f " % fft_peak_freq)
                #print("freq_identified = %f " % freq_identified)
                #print '-------------------'


                if freq_identified == self.freqIdentify[1]:
                    self.right = SignalsDetection.SIGNAL_PRIORITY
                elif freq_identified == self.freqIdentify[2]:
                    self.right = SignalsDetection.SIGNAL_SACRIFICE_FOR_PRIORITY
                else:
                    self.right = SignalsDetection.SIGNAL_A
                break

        # Decide whether LED or not (front)
        for i in range(len(BlobsFront)):
            #rospy.loginfo('[%s] Detection on the front' % (self.node_name))
            # Detection
            detected, result,freq_identified, fft_peak_freq  = self.detect_blob(BlobsFront[i],T,NIm,H,W,self.cropNormalizedFront,timestamps,result)

            # Take decision
            if detected:

                #print '-------------------'
                #print("NIm = %d " % NIm)
                #print("T = %f " % T)
                #print("fft_peak_freq = %f " % fft_peak_freq)
                #print("freq_identified = %f " % freq_identified)
                #print '-------------------'

                if freq_identified == self.freqIdentify[1]:
                    self.front = SignalsDetection.SIGNAL_PRIORITY
                elif freq_identified == self.freqIdentify[2]:
                    self.front = SignalsDetection.SIGNAL_SACRIFICE_FOR_PRIORITY
                else:
                    self.front = SignalsDetection.SIGNAL_A
                break

        # Decide whether LED or not (traffic light)
        for i in range(len(BlobsTL)):
            #rospy.loginfo('[%s] Detection of the traffic light' % (self.node_name))
            # Detection
            detected, result,freq_identified, fft_peak_freq  = self.detect_blob(BlobsTL[i],T,NIm,H,W,self.cropNormalizedTL,timestamps,result)
            # Take decision
            if detected:
                self.traffic_light = SignalsDetection.GO
                break
            else:
                self.traffic_light = SignalsDetection.STOP

        # Left bot (also UNKNOWN)
        self.left = "UNKNOWN"

        # Final time
        processing_time = time.time()-tic
        total_time      = time.time()-self.tinit

        # Publish results
        self.publish(imPublishRight,imPublishFront,imPublishTL,result)

        # Print performance
        #rospy.loginfo('[%s] Detection completed. Processing time: %.2f s. Total time:  %.2f s' %(self.node_name,processing_time,total_time))

        # Keep going
        if self.continuous:
            self.trigger = True
            self.sub_cam = rospy.Subscriber("camera_node/image/compressed",CompressedImage, self.camera_callback)

    def detect_blob(self,Blob,T,NIm,H,W,crop,timestamps,result):
        # Percentage of appearance
        apperance_percentage = (1.0*Blob['N'])/(1.0*NIm)

        # Frequency estimation based on FFT
        f              = np.arange(0.0,1.0*NIm+1.0,2.0)
        signal_f       = scipy.fftpack.fft(Blob['Signal']-np.mean(Blob['Signal']))
        y_f            = 2.0/NIm*np.abs(signal_f[:NIm/2+1])
        fft_peak_freq  = 1.0*np.argmax(y_f)/(NIm*T)
        #half_freq_dist = 0.8 #1.0*f[1]/2

        #rospy.loginfo('[%s] Appearance perc. = %s, frequency = %s' % (self.node_name, apperance_percentage, fft_peak_freq))
        freq_identified = 0
        # Take decision
        detected = False
        for i in range(len(self.freqIdentify)):
            if  (apperance_percentage < 0.8 and apperance_percentage > 0.2 and not self.useFFT) or (self.useFFT and abs(fft_peak_freq-self.freqIdentify[i]) < 0.35):
                # Decision
                detected = True
                freq_identified = self.freqIdentify[i]
                # Raw detection
                coord_norm = Vector2D(1.0*(crop[1][0]+Blob['p'][0])/W, 1.0*(crop[0][0]+Blob['p'][1])/H)
                result.detections.append(LEDDetection(rospy.Time.from_sec(timestamps[0]),rospy.Time.from_sec(timestamps[-1]),coord_norm,fft_peak_freq,'',-1,timestamps,signal_f,f,y_f))

        return detected, result, freq_identified, fft_peak_freq

    def publish(self,imRight,imFront,imTL,results):
        #  Publish image with circles
        imRightCircle_msg = self.bridge.cv2_to_imgmsg(imRight,encoding="passthrough")
        imFrontCircle_msg = self.bridge.cv2_to_imgmsg(imFront,encoding="passthrough")
        imTLCircle_msg    = self.bridge.cv2_to_imgmsg(imTL,encoding="passthrough")

        # Publish image
        self.pub_image_right.publish(imRightCircle_msg)
        self.pub_image_front.publish(imFrontCircle_msg)
        self.pub_image_TL.publish(imTLCircle_msg)

        # Publish results
        self.pub_raw_detections.publish(results)

        # Publish debug
        debug_msg = LEDDetectionDebugInfo()
        debug_msg.cell_size          = self.cell_size
        debug_msg.crop_rect_norm     = self.crop_rect_norm
        debug_msg.led_all_unfiltered = results
        debug_msg.state              = 0
        self.pub_debug.publish(debug_msg)

        # Loginfo (right)
        if self.right != SignalsDetection.NO_CAR:
            rospy.loginfo('Right: LED detected')
        else:
            rospy.loginfo('Right: No LED detected')

        # Loginfo (front)
        if self.front != SignalsDetection.NO_CAR:
            rospy.loginfo('Front: LED detected')
        else:
            rospy.loginfo('Front: No LED detected')

        # Loginfo (TL)
        if self.traffic_light == SignalsDetection.STOP:
            rospy.loginfo('[%s] Traffic Light: red' %(self.node_name))
        elif self.traffic_light == SignalsDetection.GO:
            rospy.loginfo('[%s] Traffic Light: green' %(self.node_name))
        else:
            rospy.loginfo('[%s] No traffic light' %(self.node_name))

        #Publish
        rospy.loginfo("[%s] The observed LEDs are:\n Front = %s\n Right = %s\n Traffic light state = %s" % (self.node_name, self.front, self.right, self.traffic_light))
        self.pub_detections.publish(SignalsDetection(front=self.front, right=self.right, left=self.left, traffic_light_state=self.traffic_light))

    def send_state(self, msg):
        msg.state = self.node_state
        self.pub_debug.publish(msg)

if __name__ == '__main__':
    rospy.init_node('led_detector_node',anonymous=False)
    node = LEDDetectorNode()
    rospy.spin()
