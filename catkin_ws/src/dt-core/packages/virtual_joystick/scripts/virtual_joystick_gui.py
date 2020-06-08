#!/usr/bin/python
import pygame
import time
import rospy
from sensor_msgs.msg import Joy
from duckietown_msgs.msg import BoolStamped
import os, sys
import socket
import re

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


screen_size = 300
speed_tang = 1.0
speed_norm = 1.0
time_to_wait = 10000
last_ms = 0
last_ms_p = 0
auto_restart = False
e_stop = False
dpad = None
dpad_f = None
dpad_r = None
dpad_b = None
dpad_l = None
dpad_estop = None
dpad_estop_big = None
estop_last_time = time.time()
estop_deadzone_secs = 0.5


# TODO: This should be a class

def loop():
    global last_ms, time_to_wait, last_ms_p, e_stop, estop_last_time
    veh_standing = True

    while True:
        ms_now = int(round(time.time() * 1000))
        if ms_now - last_ms > time_to_wait:
            # query rosmaster status, which will raise socket.error when failure
            rospy.get_master().getSystemState()
            # end-of-checking
            last_ms = ms_now

        # add dpad to screen
        screen.blit(dpad, (0,0))

        # draw e-stop signal
        if e_stop:
            screen.blit(dpad_estop, (0, 0))

        # prepare message
        msg = Joy()
        msg.header.seq = 0
        msg.header.stamp.secs = 0
        msg.header.stamp.nsecs = 0
        msg.header.frame_id = ''
        msg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        force_joy_publish = False

        # obtain pressed keys
        keys = pygame.key.get_pressed()

        # START CHECKING KEYS #

        # drive left
        if keys[pygame.K_LEFT]:
            if e_stop:
                screen.blit(dpad_estop_big, (0, 0))
            else:
                screen.blit(dpad_l, (0,0))
            msg.axes[3] += speed_norm

        # drive right
        if keys[pygame.K_RIGHT]:
            if e_stop:
                screen.blit(dpad_estop_big, (0, 0))
            else:
                screen.blit(dpad_r, (0, 0))
            msg.axes[3] -= speed_norm

        # drive forward
        if keys[pygame.K_UP]:
            if e_stop:
                screen.blit(dpad_estop_big, (0, 0))
            else:
                screen.blit(dpad_f, (0, 0))
            msg.axes[1] += speed_tang

        # drive backwards
        if keys[pygame.K_DOWN]:
            if e_stop:
                screen.blit(dpad_estop_big, (0, 0))
            else:
                screen.blit(dpad_b, (0, 0))
            msg.axes[1] -= speed_tang

        # TODO: intersection_go?
        if keys[pygame.K_p]:
            msg_int = BoolStamped()
            msg_int.data = True
            pub_int.publish(msg_int)

        # activate line-following
        if keys[pygame.K_a]:
            msg.buttons[7] = 1

        # stop line-following
        if keys[pygame.K_s]:
            msg.buttons[6] = 1

        # toggle anti-instagram
        if keys[pygame.K_i]:
            msg.buttons[3] = 1

        # toggle emergency stop
        if keys[pygame.K_e] and (time.time() - estop_last_time > estop_deadzone_secs):
            msg.buttons[3] = 1
            estop_last_time = time.time()
            force_joy_publish = True

        # key/action for quitting the program

        # check if window exit button [x] was hit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # quit program
        if keys[pygame.K_q]:
            pygame.quit()

        # END CHECKING KEYS #

        # refresh screen
        pygame.display.flip()

        # check for any input commands
        stands = (sum(map(abs, msg.axes)) == 0 and sum(map(abs, msg.buttons)) == 0)
        if not stands:
            veh_standing = False

        # publish message
        if (not veh_standing) or force_joy_publish:
            pub_joystick.publish(msg)

        # adjust veh_standing such that when vehicle stands still, at least
        # one last publishment was sent to the bot. That's why this adjustment
        # is made after the publishment of the message
        if stands:
            veh_standing = True

        time.sleep(0.03)

        # obtain next key list
        pygame.event.pump()


# prepare size and rotations of dpad and dpad_pressed
def prepare_dpad():
    global dpad, dpad_f, dpad_r, dpad_b, dpad_l, dpad_estop, dpad_estop_big
    script_path = os.path.dirname(__file__)
    script_path = (script_path + "/") if script_path else ""
    # draw d-pad
    dpad = pygame.image.load(script_path + "../images/d-pad.png")
    dpad = pygame.transform.scale(dpad, (screen_size, screen_size))
    # draw pressed-arrows
    dpad_pressed = pygame.image.load(script_path + "../images/d-pad-pressed.png")
    dpad_pressed = pygame.transform.scale(dpad_pressed, (screen_size, screen_size))
    dpad_f = dpad_pressed
    dpad_r = pygame.transform.rotate(dpad_pressed, 270)
    dpad_b = pygame.transform.rotate(dpad_pressed, 180)
    dpad_l = pygame.transform.rotate(dpad_pressed, 90)
    # draw E-Stop
    estop_img = pygame.image.load(script_path + "../images/d-e-stop.png")
    estop_big_img = pygame.image.load(script_path + "../images/d-e-stop-big.png")
    dpad_estop = pygame.transform.scale(estop_img, (screen_size, screen_size))
    dpad_estop_big = pygame.transform.scale(estop_big_img, (screen_size, screen_size))


def cbEStop(estop_msg):
    """
    Callback that process the received :obj:`BoolStamped` messages.

    Args:
        estop_msg (:obj:`BoolStamped`): the emergency_stop message to process.
    """
    global e_stop
    e_stop = estop_msg.data
    if e_stop:
        # add dpad to screen
        screen.blit(dpad, (0, 0))
        # add e-stop to screen
        screen.blit(dpad_estop, (0, 0))
        # refresh screen
        pygame.display.flip()


# Hint which is print at startup in console
def print_hint():
    print("\n\n\n")
    print("Virtual Joystick for your Duckiebot")
    print("-----------------------------------")
    print("\n")
    print("[ARROW_KEYS]:    Use them to steer your Duckiebot")
    print("         [q]:    Quit the program")
    print("         [a]:    Start lane-following a.k.a. autopilot")
    print("         [s]:    Stop lane-following")
    print("         [i]:    Toggle anti-instagram")
    print("         [e]:    Toggle emergency stop")
    print("\n")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception("No hostname specified!")
    else:
        veh_name = sys.argv[1]

    veh_no = re.sub("\D", "", veh_name)
    main_letter = veh_name[0]

    # prepare pygame
    pygame.init()

    file_dir = os.path.dirname(__file__)
    file_dir = (file_dir + "/") if file_dir else ""
    logo = pygame.image.load(file_dir + "../images/logo.png")

    pygame.display.set_icon(logo)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption(veh_name)

    prepare_dpad()

    # prepare ROS node
    rospy.init_node('virtual_joy', anonymous=False)

    # prepare ROS subscribers
    sub_estop = rospy.Subscriber("~emergency_stop", BoolStamped, cbEStop, queue_size=1)

    # prepare ROS publisher
    pub_joystick = rospy.Publisher("~joy", Joy, queue_size=1)
    pub_int = rospy.Publisher("~intersection_go", BoolStamped, queue_size=1)

    # print the hint
    print_hint()

    try:
        # start the main loop
        loop()
    except socket.error:
        print("Error starting main loop in virtual joystick gui")
