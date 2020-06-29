// Auto-generated. Do not edit!

// (in-package duckietown_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class ToFStamped {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.error = null;
      this.distance = null;
      this.confidence = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('error')) {
        this.error = initObj.error
      }
      else {
        this.error = 0;
      }
      if (initObj.hasOwnProperty('distance')) {
        this.distance = initObj.distance
      }
      else {
        this.distance = 0;
      }
      if (initObj.hasOwnProperty('confidence')) {
        this.confidence = initObj.confidence
      }
      else {
        this.confidence = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ToFStamped
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [error]
    bufferOffset = _serializer.int8(obj.error, buffer, bufferOffset);
    // Serialize message field [distance]
    bufferOffset = _serializer.int16(obj.distance, buffer, bufferOffset);
    // Serialize message field [confidence]
    bufferOffset = _serializer.int16(obj.confidence, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ToFStamped
    let len;
    let data = new ToFStamped(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [error]
    data.error = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [distance]
    data.distance = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [confidence]
    data.confidence = _deserializer.int16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 5;
  }

  static datatype() {
    // Returns string type for a message object
    return 'duckietown_msgs/ToFStamped';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '803258046ff4b4331a4d8f4f3f180cc9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    # There is no error. `distance` is valid.
    int8 NO_ERROR = 0
    # Target is too close to the sensor
    int8 ERROR_NEAR = 1
    # Target is too far from the sensor (> 2047mm)
    int8 ERROR_FAR = 2
    # Other general error
    int8 ERROR_OTHER = 3
    
    int8 error       # One of NO_ERROR, ERROR_NEAR, ERROR_FAR, or ERROR_OTHER
    int16 distance   # Distance in mm. Only valid if error == NO_ERROR
    int16 confidence # An abstract "confidence" measurement that is not well-defined in the RFD77402 datasheet
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ToFStamped(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.error !== undefined) {
      resolved.error = msg.error;
    }
    else {
      resolved.error = 0
    }

    if (msg.distance !== undefined) {
      resolved.distance = msg.distance;
    }
    else {
      resolved.distance = 0
    }

    if (msg.confidence !== undefined) {
      resolved.confidence = msg.confidence;
    }
    else {
      resolved.confidence = 0
    }

    return resolved;
    }
};

// Constants for message
ToFStamped.Constants = {
  NO_ERROR: 0,
  ERROR_NEAR: 1,
  ERROR_FAR: 2,
  ERROR_OTHER: 3,
}

module.exports = ToFStamped;
