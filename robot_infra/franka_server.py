"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
import flask
from flask import Flask, request, jsonify
import rospy
import numpy as np
import json
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, StopActionGoal
from franka_msgs.msg import ErrorRecoveryActionGoal
from franka_msgs.msg import FrankaState
from franka_msgs.msg import ZeroJacobian
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
import geometry_msgs.msg as geom_msg
import time
import select
from multiprocessing import Process
import subprocess
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import os
import cv2
import sys
from dynamic_reconfigure.client import Client

IMSIZE = 256

def get_cameras():
    cams = []
    for i in range(10):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            cams.append(i)
        cam.release()
    print("cameras", cams)
    return cams


app = Flask(__name__)
## IF YOU ARE NOT IN IRIS CHANGES THIS FOR YOUR NUC HOME DIR AND ROBOT IP
ROS_WS = "/home/jianlan/code/catkin_ws"
ROBOT_IP = "172.16.0.2"


class Launcher:
    """Handles the starting and stopping of the impedence controller
    (as well as backup) joint recovery policy."""

    def __init__(self):
        self.gripper_command = outputMsg.Robotiq2FGripper_robot_output()

    def start_impedence(self):
        ## Launches the impedence controller
        self.imp = subprocess.Popen(
            [
                "roslaunch",
                ROS_WS + "/scripts/impedence.launch",
                "robot_ip:=" + ROBOT_IP,
                "load_gripper:=false",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(5)

    """Update the gripper command according to the character entered by the user."""
    def generate_gripper_command(self, char, command):
        if char == "a":
            command = outputMsg.Robotiq2FGripper_robot_output()
            command.rACT = 1
            command.rGTO = 1
            command.rSP = 255
            command.rFR = 150

        if char == "r":
            command = outputMsg.Robotiq2FGripper_robot_output()
            command.rACT = 0

        if char == "c":
            command.rPR = 255

        if char == "o":
            command.rPR = 0

        # If the command entered is a int, assign this value to rPR 
        # (i.e., move to this position)
        try:
            command.rPR = int(char)
            if command.rPR > 255:
                command.rPR = 255
            if command.rPR < 0:
                command.rPR = 0
        except ValueError:
            pass
        return command

    def stop_impedence(self):
        ## Stops the impedence controller
        self.imp.terminate()
        time.sleep(1)

    def set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        self.force = np.array(list(msg.O_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.O_F_ext_hat_K)[3:])
        self.vel = self.jacobian @ self.dq

    def set_jacobian(self, msg):
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # import pdb; pdb.set_trace()
        # First Stop Impedence
        try:
            self.stop_impedence()
            
        except:
            print("Impedence Not Running")

        ## Launch joint controller reset
        clear()
        self.j = subprocess.Popen(
            [
                "roslaunch",
                ROS_WS + "/scripts/joint.launch",
                "robot_ip:=" + ROBOT_IP,
                "load_gripper:=false",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        clear()
        count = 0
        time.sleep(1)
        while not np.allclose(np.array([0, 0, 0, -1.9, -0, 2, 0]) - np.array(self.q), 0,
                        atol=1e-2, rtol=1e-2):
            time.sleep(1)
            count += 1
            if count > 100:
                print('TIMEOUT')
                break

        print("RESET DONE")
        self.j.terminate()
        time.sleep(1)
        clear()
        print("KILLED JOINT RESET", self.pos)
        self.start_impedence()
        print("IMPEDENCE STARTED")


"""Starts Impedence controller"""
l = Launcher()
l.start_impedence()

## Defines the ros topics to publish to
# rospy.init_node("equilibrium_pose_node")
rospy.init_node("franka_control_api")
gripperpub = rospy.Publisher(
    "Robotiq2FGripperRobotOutput", outputMsg.Robotiq2FGripper_robot_output, queue_size=1
)
gripperpub.publish(l.gripper_command) # init reset gripper
time.sleep(1)

eepub = rospy.Publisher(
    "/cartesian_impedance_example_controller/equilibrium_pose",
    geom_msg.PoseStamped,
    queue_size=10,
)
resetpub = rospy.Publisher(
    "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
)
state_sub = rospy.Subscriber(
    "franka_state_controller/franka_states", FrankaState, l.set_currpos
)
jacobian_sub = rospy.Subscriber(
    "/cartesian_impedance_example_controller/franka_jacobian",
    ZeroJacobian,
    l.set_jacobian,
)
# l.reset_joint()
client = Client("cartesian_impedance_example_controllerdynamic_reconfigure_compliance_param_node")

## Route for Starting Impedence
@app.route("/startimp", methods=["POST"])
def si():
    clear()
    l.start_impedence()
    return "Started Impedence"


## Route for Stopping Impedence
@app.route("/stopimp", methods=["POST"])
def sti():
    l.stop_impedence()
    return "Stopped Impedence"


## Route for Getting Pose
@app.route("/getpos", methods=["POST"])
def gp():
    return jsonify({"pose": np.array(l.pos).tolist()})


@app.route("/getvel", methods=["POST"])
def gv():
    return jsonify({"vel": np.array(l.vel).tolist()})


@app.route("/getforce", methods=["POST"])
def gf():
    return jsonify({"force": np.array(l.force).tolist()})


@app.route("/gettorque", methods=["POST"])
def gt():
    return jsonify({"torque": np.array(l.torque).tolist()})


@app.route("/getq", methods=["POST"])
def gq():
    return jsonify({"q": np.array(l.q).tolist()})


@app.route("/getdq", methods=["POST"])
def gdq():
    return jsonify({"dq": np.array(l.dq).tolist()})


@app.route("/getjacobian", methods=["POST"])
def gj():
    return jsonify({"jacobian": np.array(l.jacobian).tolist()})

## Route for Running Joint Reset
@app.route("/jointreset", methods=["POST"])
def jr():
    msg = ErrorRecoveryActionGoal()
    resetpub.publish(msg)
    l.reset_joint()
    return "Reset Joint"

##Route for Activating the Gripper
@app.route("/activate_gripper", methods=["POST"])
def activate_gripper():
    print("activate gripper")
    l.gripper_command = l.generate_gripper_command("a", l.gripper_command)
    gripperpub.publish(l.gripper_command)
    return "Activated"

## Route for Resetting the Gripper. It will reset and activate the gripper
@app.route("/reset_gripper", methods=["POST"])
def reset_gripper():
    print("reset gripper")
    l.gripper_command = l.generate_gripper_command("r", l.gripper_command)
    gripperpub.publish(l.gripper_command)
    l.gripper_command = l.generate_gripper_command("a", l.gripper_command)
    gripperpub.publish(l.gripper_command)
    return "Reset"

## Route for Opening the Gripper
@app.route("/open", methods=["POST"])
def open():
    print("open")
    l.gripper_command = l.generate_gripper_command("o", l.gripper_command)
    gripperpub.publish(l.gripper_command)
    return "Opened"

## Route for Closing the Gripper
@app.route("/close", methods=["POST"])
def close():
    print("close")
    l.gripper_command = l.generate_gripper_command("c", l.gripper_command)
    gripperpub.publish(l.gripper_command)
    return "Closed"

## Route for moving the gripper
@app.route("/move", methods=["POST"])
def move_gripper():
    gripper_pos = request.json
    pos = int(gripper_pos["gripper_pos"] * 255) #convert from 0-1 to 0-255
    print(f"move gripper to {pos}")
    l.gripper_command = l.generate_gripper_command(pos, l.gripper_command)
    gripperpub.publish(l.gripper_command)
    return "Moved Gripper"

## Route for Clearing Errors (Communcation constraints, etc.)
@app.route("/clearerr", methods=["POST"])
def clear():
    msg = ErrorRecoveryActionGoal()
    resetpub.publish(msg)
    return "Clear"

## Route for Sending a pose command
@app.route("/pose", methods=["POST"])
def pose():
    pos = request.json
    pos = np.array(pos["arr"])
    print("Moving to", pos)
    msg = geom_msg.PoseStamped()
    msg.header.frame_id = "0"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position = geom_msg.Point(pos[0], pos[1], pos[2])
    msg.pose.orientation = geom_msg.Quaternion(pos[3], pos[4], pos[5], pos[6])
    eepub.publish(msg)
    return "Moved"

@app.route("/getstate", methods=["POST"])
def gs():
    return jsonify({"pose": np.array(l.pos).tolist(),
                    "vel": np.array(l.vel).tolist(),
                    "force": np.array(l.force).tolist(),
                    "torque": np.array(l.torque).tolist(),
                    "q": np.array(l.q).tolist(),
                    "dq": np.array(l.dq).tolist(),
                    "jacobian": np.array(l.jacobian).tolist()})
# PCB
# @app.route("/pcb_compliance_mode", methods=["POST"])
# def pcb_compliance_mode():
#     client.update_configuration({"translational_stiffness": 3000})
#     client.update_configuration({"translational_damping": 180})
#     client.update_configuration({"rotational_stiffness": 150})
#     client.update_configuration({"rotational_damping": 7})
#     client.update_configuration({"translational_clip_neg_x": 0.002})
#     client.update_configuration({"translational_clip_neg_y": 0.001})
#     client.update_configuration({"translational_clip_neg_z": 0.002})
#     client.update_configuration({"translational_clip_x": 0.0015})
#     client.update_configuration({"translational_clip_y": 0.0005})
#     client.update_configuration({"translational_clip_z": 0.0014})
#     client.update_configuration({"rotational_clip_neg_x": 0.015})
#     client.update_configuration({"rotational_clip_neg_y": 0.002})
#     client.update_configuration({"rotational_clip_neg_z": 0.005})
#     client.update_configuration({"rotational_clip_x": 0.016})
#     client.update_configuration({"rotational_clip_y": 0.002})
#     client.update_configuration({"rotational_clip_z": 0.005})
#     client.update_configuration({"translational_Ki": 0})
#     client.update_configuration({"rotational_Ki": 0})
#     return "pcb compliance Mode"    

# Peg
@app.route("/peg_compliance_mode", methods=["POST"])
def peg_compliance_mode():
    client.update_configuration({"translational_stiffness": 2000})
    client.update_configuration({"translational_damping": 89})
    client.update_configuration({"rotational_stiffness": 150})
    client.update_configuration({"rotational_damping": 7})
    client.update_configuration({"translational_Ki": 30})
    client.update_configuration({"translational_clip_x": 0.005})
    client.update_configuration({"translational_clip_y": 0.005})
    client.update_configuration({"translational_clip_z": 0.005})
    client.update_configuration({"translational_clip_neg_x": 0.005})
    client.update_configuration({"translational_clip_neg_y": 0.005})
    client.update_configuration({"translational_clip_neg_z": 0.005})
    client.update_configuration({"rotational_clip_x": 0.05})
    client.update_configuration({"rotational_clip_y": 0.05})
    client.update_configuration({"rotational_clip_z": 0.05})
    client.update_configuration({"rotational_clip_neg_x": 0.05})
    client.update_configuration({"rotational_clip_neg_y": 0.05})
    client.update_configuration({"rotational_clip_neg_z": 0.05})
    client.update_configuration({"rotational_Ki": 0})
    return "peg compliance Mode"


@app.route("/precision_mode", methods=["POST"])
def precision_mode():
    client.update_configuration({"translational_stiffness": 2000})
    client.update_configuration({"translational_damping": 89})
    client.update_configuration({"rotational_stiffness": 250})
    client.update_configuration({"rotational_damping": 9})
    client.update_configuration({"translational_Ki": 30})
    client.update_configuration({"translational_clip_x": 0.1})
    client.update_configuration({"translational_clip_y": 0.1})
    client.update_configuration({"translational_clip_z": 0.1})
    client.update_configuration({"translational_clip_neg_x": 0.1})
    client.update_configuration({"translational_clip_neg_y": 0.1})
    client.update_configuration({"translational_clip_neg_z": 0.1})
    client.update_configuration({"rotational_clip_x": 0.1})
    client.update_configuration({"rotational_clip_y": 0.1})
    client.update_configuration({"rotational_clip_z": 0.1})
    client.update_configuration({"rotational_clip_neg_x": 0.1})
    client.update_configuration({"rotational_clip_neg_y": 0.1})
    client.update_configuration({"rotational_clip_neg_z": 0.1})
    client.update_configuration({"rotational_Ki": 5})
    return "precision Mode"

# cameras = get_cameras() #range(10) #[1]
import threading

class Camera:
    def __init__(self):
        all_c = get_cameras()
        self.cams = []
        for c in all_c:
            # newcam = cv2.VideoCapture(c)
            # newcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cams.append(cv2.VideoCapture(c))
        print("CREATED CAM")
        self.imsize = 256

        # self.read_and_save()
        self.imthread = threading.Thread(
            target=self.read_and_save, name="Image Capture"
        )
        self.imthread.start()
        time.sleep(10)

        self.get()

    def read_and_save(self):
        while True:
            frames = []
            for c in self.cams:
                r = False
                while not r:
                    r, frame = c.read()
                    # print("Read")
                frame = cv2.resize(
                    frame, (self.imsize, self.imsize), interpolation=cv2.INTER_AREA
                )
                frames.append(frame)
            self.lastob = np.concatenate(frames, axis=0)
            # print("LAST OB WROTE")

    def get(self):
        print(self.imthread.isAlive())
        return self.lastob

# cam = Camera()

## Route for Stopping Impedence
@app.route("/image", methods=["POST"])
def img():
    obs = cam.get()
    response = flask.make_response(obs.tobytes())
    response.headers.set("Content-Type", "application/octet-stream")
    t6 = time.time()
    return response


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0")
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        l.stop_impedence()
        sys.exit()
