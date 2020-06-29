#!/usr/bin/env python2
import rospy

from sensor_msgs.msg import Joy


# Button List index of joy.buttons array:
#   0: A
#   1: B
#   2: X
#   3: Y
#   4: Left Back
#   5: Right Back
#   6: Back
#   7: Start
#   8: Logitek
#   9: Left joystick
#   10: Right joystick


# TODO: This should be a class

def keyCatcher():
    rospy.init_node('joy-cli')
    pub = rospy.Publisher('~joy', Joy, queue_size=1)

    while not rospy.is_shutdown():
        axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        direction = raw_input(
            'Enter direction [a,w,s,d] to move; '
            '[l] to start lane following; '
            '[q] to stop lane following; '
            '[e] to toggle emergency stop; '
            'then press enter to execute --> ')
        if direction == 'w':
            axes[1] = 1.0
        elif direction == 's':
            axes[1] = -1.0
        elif direction == 'd':
            axes[3] = -1.0
        elif direction == 'a':
            axes[3] = 1.0
        elif direction == 'l':
            buttons[7] = 1
        elif direction == 'q':
            buttons[6] = 1
        elif direction == 'e':
            buttons[3] = 1
        # publish joy message
        msg = Joy(header=None, axes=axes, buttons=buttons)
        pub.publish(msg)
        rospy.sleep(0.5)


if __name__ == '__main__':
    try:
        keyCatcher()
    except rospy.ROSInterruptException:
        pass

