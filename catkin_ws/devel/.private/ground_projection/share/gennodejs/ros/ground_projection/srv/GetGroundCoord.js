// Auto-generated. Do not edit!

// (in-package ground_projection.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let duckietown_msgs = _finder('duckietown_msgs');

//-----------------------------------------------------------

let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class GetGroundCoordRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.uv = null;
    }
    else {
      if (initObj.hasOwnProperty('uv')) {
        this.uv = initObj.uv
      }
      else {
        this.uv = new duckietown_msgs.msg.Pixel();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetGroundCoordRequest
    // Serialize message field [uv]
    bufferOffset = duckietown_msgs.msg.Pixel.serialize(obj.uv, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetGroundCoordRequest
    let len;
    let data = new GetGroundCoordRequest(null);
    // Deserialize message field [uv]
    data.uv = duckietown_msgs.msg.Pixel.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'ground_projection/GetGroundCoordRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'fba398636bea4394b5da7cf9599a6d3c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    duckietown_msgs/Pixel uv
    
    
    ================================================================================
    MSG: duckietown_msgs/Pixel
    int32 u
    int32 v
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetGroundCoordRequest(null);
    if (msg.uv !== undefined) {
      resolved.uv = duckietown_msgs.msg.Pixel.Resolve(msg.uv)
    }
    else {
      resolved.uv = new duckietown_msgs.msg.Pixel()
    }

    return resolved;
    }
};

class GetGroundCoordResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.gp = null;
    }
    else {
      if (initObj.hasOwnProperty('gp')) {
        this.gp = initObj.gp
      }
      else {
        this.gp = new geometry_msgs.msg.Point();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetGroundCoordResponse
    // Serialize message field [gp]
    bufferOffset = geometry_msgs.msg.Point.serialize(obj.gp, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetGroundCoordResponse
    let len;
    let data = new GetGroundCoordResponse(null);
    // Deserialize message field [gp]
    data.gp = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'ground_projection/GetGroundCoordResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'cb06b1906fc6f1f5910a7d2012f835c0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    geometry_msgs/Point gp
    
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetGroundCoordResponse(null);
    if (msg.gp !== undefined) {
      resolved.gp = geometry_msgs.msg.Point.Resolve(msg.gp)
    }
    else {
      resolved.gp = new geometry_msgs.msg.Point()
    }

    return resolved;
    }
};

module.exports = {
  Request: GetGroundCoordRequest,
  Response: GetGroundCoordResponse,
  md5sum() { return '4a593ebeb349392dda67c9ac9b35b490'; },
  datatype() { return 'ground_projection/GetGroundCoord'; }
};
