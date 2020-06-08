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

class LEDPattern {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.color_list = null;
      this.color_mask = null;
      this.frequency = null;
      this.frequency_mask = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('color_list')) {
        this.color_list = initObj.color_list
      }
      else {
        this.color_list = [];
      }
      if (initObj.hasOwnProperty('color_mask')) {
        this.color_mask = initObj.color_mask
      }
      else {
        this.color_mask = [];
      }
      if (initObj.hasOwnProperty('frequency')) {
        this.frequency = initObj.frequency
      }
      else {
        this.frequency = 0.0;
      }
      if (initObj.hasOwnProperty('frequency_mask')) {
        this.frequency_mask = initObj.frequency_mask
      }
      else {
        this.frequency_mask = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type LEDPattern
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [color_list]
    bufferOffset = _arraySerializer.string(obj.color_list, buffer, bufferOffset, null);
    // Serialize message field [color_mask]
    bufferOffset = _arraySerializer.int8(obj.color_mask, buffer, bufferOffset, null);
    // Serialize message field [frequency]
    bufferOffset = _serializer.float32(obj.frequency, buffer, bufferOffset);
    // Serialize message field [frequency_mask]
    bufferOffset = _arraySerializer.int8(obj.frequency_mask, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type LEDPattern
    let len;
    let data = new LEDPattern(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [color_list]
    data.color_list = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [color_mask]
    data.color_mask = _arrayDeserializer.int8(buffer, bufferOffset, null)
    // Deserialize message field [frequency]
    data.frequency = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [frequency_mask]
    data.frequency_mask = _arrayDeserializer.int8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.color_list.forEach((val) => {
      length += 4 + val.length;
    });
    length += object.color_mask.length;
    length += object.frequency_mask.length;
    return length + 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'duckietown_msgs/LEDPattern';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4ee02b2a0be46c1e87c03d4456ee3012';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    string[]  color_list
    int8[]    color_mask
    float32   frequency
    int8[]    frequency_mask
    
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
    const resolved = new LEDPattern(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.color_list !== undefined) {
      resolved.color_list = msg.color_list;
    }
    else {
      resolved.color_list = []
    }

    if (msg.color_mask !== undefined) {
      resolved.color_mask = msg.color_mask;
    }
    else {
      resolved.color_mask = []
    }

    if (msg.frequency !== undefined) {
      resolved.frequency = msg.frequency;
    }
    else {
      resolved.frequency = 0.0
    }

    if (msg.frequency_mask !== undefined) {
      resolved.frequency_mask = msg.frequency_mask;
    }
    else {
      resolved.frequency_mask = []
    }

    return resolved;
    }
};

module.exports = LEDPattern;
