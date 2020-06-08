// Auto-generated. Do not edit!

// (in-package ground_projection.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

let duckietown_msgs = _finder('duckietown_msgs');

//-----------------------------------------------------------

class GetImageCoordRequest {
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
    // Serializes a message object of type GetImageCoordRequest
    // Serialize message field [gp]
    bufferOffset = geometry_msgs.msg.Point.serialize(obj.gp, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetImageCoordRequest
    let len;
    let data = new GetImageCoordRequest(null);
    // Deserialize message field [gp]
    data.gp = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'ground_projection/GetImageCoordRequest';
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
    const resolved = new GetImageCoordRequest(null);
    if (msg.gp !== undefined) {
      resolved.gp = geometry_msgs.msg.Point.Resolve(msg.gp)
    }
    else {
      resolved.gp = new geometry_msgs.msg.Point()
    }

    return resolved;
    }
};

class GetImageCoordResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.normalized_uv = null;
    }
    else {
      if (initObj.hasOwnProperty('normalized_uv')) {
        this.normalized_uv = initObj.normalized_uv
      }
      else {
        this.normalized_uv = new duckietown_msgs.msg.Vector2D();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetImageCoordResponse
    // Serialize message field [normalized_uv]
    bufferOffset = duckietown_msgs.msg.Vector2D.serialize(obj.normalized_uv, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetImageCoordResponse
    let len;
    let data = new GetImageCoordResponse(null);
    // Deserialize message field [normalized_uv]
    data.normalized_uv = duckietown_msgs.msg.Vector2D.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'ground_projection/GetImageCoordResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7e079b9787496ba75117334836e96c45';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    
    duckietown_msgs/Vector2D normalized_uv
    
    ================================================================================
    MSG: duckietown_msgs/Vector2D
    float32 x
    float32 y
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetImageCoordResponse(null);
    if (msg.normalized_uv !== undefined) {
      resolved.normalized_uv = duckietown_msgs.msg.Vector2D.Resolve(msg.normalized_uv)
    }
    else {
      resolved.normalized_uv = new duckietown_msgs.msg.Vector2D()
    }

    return resolved;
    }
};

module.exports = {
  Request: GetImageCoordRequest,
  Response: GetImageCoordResponse,
  md5sum() { return '590f09a60b9d5d3e1b4384278b6e2b2f'; },
  datatype() { return 'ground_projection/GetImageCoord'; }
};
