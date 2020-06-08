
"use strict";

let LEDInterpreter = require('./LEDInterpreter.js');
let ObstacleProjectedDetection = require('./ObstacleProjectedDetection.js');
let CarControl = require('./CarControl.js');
let EncoderStamped = require('./EncoderStamped.js');
let BoolStamped = require('./BoolStamped.js');
let SegmentList = require('./SegmentList.js');
let Segment = require('./Segment.js');
let LEDDetectionDebugInfo = require('./LEDDetectionDebugInfo.js');
let ParamTuner = require('./ParamTuner.js');
let IntersectionPose = require('./IntersectionPose.js');
let WheelsCmdStamped = require('./WheelsCmdStamped.js');
let SceneSegments = require('./SceneSegments.js');
let VehiclePose = require('./VehiclePose.js');
let StopLineReading = require('./StopLineReading.js');
let DroneControl = require('./DroneControl.js');
let WheelsCmd = require('./WheelsCmd.js');
let ObstacleImageDetection = require('./ObstacleImageDetection.js');
let ThetaDotSample = require('./ThetaDotSample.js');
let LightSensor = require('./LightSensor.js');
let AntiInstagramTransform_CB = require('./AntiInstagramTransform_CB.js');
let DuckiebotLED = require('./DuckiebotLED.js');
let AntiInstagramTransform = require('./AntiInstagramTransform.js');
let DuckieSensor = require('./DuckieSensor.js');
let Pose2DStamped = require('./Pose2DStamped.js');
let VehicleCorners = require('./VehicleCorners.js');
let ToFStamped = require('./ToFStamped.js');
let ObstacleProjectedDetectionList = require('./ObstacleProjectedDetectionList.js');
let AprilTagExtended = require('./AprilTagExtended.js');
let Rect = require('./Rect.js');
let LEDDetection = require('./LEDDetection.js');
let IntersectionPoseImgDebug = require('./IntersectionPoseImgDebug.js');
let KinematicsParameters = require('./KinematicsParameters.js');
let TagInfo = require('./TagInfo.js');
let StreetNames = require('./StreetNames.js');
let DroneMode = require('./DroneMode.js');
let MaintenanceState = require('./MaintenanceState.js');
let AprilTagsWithInfos = require('./AprilTagsWithInfos.js');
let Rects = require('./Rects.js');
let KinematicsWeights = require('./KinematicsWeights.js');
let StreetNameDetection = require('./StreetNameDetection.js');
let AntiInstagramHealth = require('./AntiInstagramHealth.js');
let Vector2D = require('./Vector2D.js');
let AprilTagDetection = require('./AprilTagDetection.js');
let FSMState = require('./FSMState.js');
let CoordinationSignal = require('./CoordinationSignal.js');
let LEDDetectionArray = require('./LEDDetectionArray.js');
let AprilTagDetectionArray = require('./AprilTagDetectionArray.js');
let LEDPattern = require('./LEDPattern.js');
let WheelsCmdDBV2Stamped = require('./WheelsCmdDBV2Stamped.js');
let ObstacleImageDetectionList = require('./ObstacleImageDetectionList.js');
let LanePose = require('./LanePose.js');
let CoordinationClearance = require('./CoordinationClearance.js');
let SignalsDetectionETHZ17 = require('./SignalsDetectionETHZ17.js');
let SourceTargetNodes = require('./SourceTargetNodes.js');
let TurnIDandType = require('./TurnIDandType.js');
let Pixel = require('./Pixel.js');
let ObstacleType = require('./ObstacleType.js');
let LineFollowerStamped = require('./LineFollowerStamped.js');
let Trajectory = require('./Trajectory.js');
let Twist2DStamped = require('./Twist2DStamped.js');
let SignalsDetection = require('./SignalsDetection.js');
let IntersectionPoseImg = require('./IntersectionPoseImg.js');
let Vsample = require('./Vsample.js');

module.exports = {
  LEDInterpreter: LEDInterpreter,
  ObstacleProjectedDetection: ObstacleProjectedDetection,
  CarControl: CarControl,
  EncoderStamped: EncoderStamped,
  BoolStamped: BoolStamped,
  SegmentList: SegmentList,
  Segment: Segment,
  LEDDetectionDebugInfo: LEDDetectionDebugInfo,
  ParamTuner: ParamTuner,
  IntersectionPose: IntersectionPose,
  WheelsCmdStamped: WheelsCmdStamped,
  SceneSegments: SceneSegments,
  VehiclePose: VehiclePose,
  StopLineReading: StopLineReading,
  DroneControl: DroneControl,
  WheelsCmd: WheelsCmd,
  ObstacleImageDetection: ObstacleImageDetection,
  ThetaDotSample: ThetaDotSample,
  LightSensor: LightSensor,
  AntiInstagramTransform_CB: AntiInstagramTransform_CB,
  DuckiebotLED: DuckiebotLED,
  AntiInstagramTransform: AntiInstagramTransform,
  DuckieSensor: DuckieSensor,
  Pose2DStamped: Pose2DStamped,
  VehicleCorners: VehicleCorners,
  ToFStamped: ToFStamped,
  ObstacleProjectedDetectionList: ObstacleProjectedDetectionList,
  AprilTagExtended: AprilTagExtended,
  Rect: Rect,
  LEDDetection: LEDDetection,
  IntersectionPoseImgDebug: IntersectionPoseImgDebug,
  KinematicsParameters: KinematicsParameters,
  TagInfo: TagInfo,
  StreetNames: StreetNames,
  DroneMode: DroneMode,
  MaintenanceState: MaintenanceState,
  AprilTagsWithInfos: AprilTagsWithInfos,
  Rects: Rects,
  KinematicsWeights: KinematicsWeights,
  StreetNameDetection: StreetNameDetection,
  AntiInstagramHealth: AntiInstagramHealth,
  Vector2D: Vector2D,
  AprilTagDetection: AprilTagDetection,
  FSMState: FSMState,
  CoordinationSignal: CoordinationSignal,
  LEDDetectionArray: LEDDetectionArray,
  AprilTagDetectionArray: AprilTagDetectionArray,
  LEDPattern: LEDPattern,
  WheelsCmdDBV2Stamped: WheelsCmdDBV2Stamped,
  ObstacleImageDetectionList: ObstacleImageDetectionList,
  LanePose: LanePose,
  CoordinationClearance: CoordinationClearance,
  SignalsDetectionETHZ17: SignalsDetectionETHZ17,
  SourceTargetNodes: SourceTargetNodes,
  TurnIDandType: TurnIDandType,
  Pixel: Pixel,
  ObstacleType: ObstacleType,
  LineFollowerStamped: LineFollowerStamped,
  Trajectory: Trajectory,
  Twist2DStamped: Twist2DStamped,
  SignalsDetection: SignalsDetection,
  IntersectionPoseImg: IntersectionPoseImg,
  Vsample: Vsample,
};
