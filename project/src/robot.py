import numpy as np
import pybullet as p
import itertools
from pdb import set_trace
import math


class Robot():
    """
    The class is the interface to a single robot
    """

    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()

        # No friction between bbody and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)

        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []

    def reset(self):
        """
        Moves the robot back to its initial position
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))

    def set_wheel_velocity(self, vel):
        """
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel)
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
                                    targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]

    def get_messages(self):
        """
        returns a list of received messages, each element of the list is a tuple (a,b)
        where a= id of the sending robot and b= message (can be any object, list, etc chosen by user)
        Note that the message will only be received if the robot is a neighbor (i.e. is close enough)
        """
        return self.messages_received

    def send_message(self, robot_id, message):
        """
        sends a message to robot with id number robot_id, the message can be any object, list, etc
        """
        self.messages_to_send.append([robot_id, message])

    def get_neighbors(self):
        """
        returns a list of neighbors (i.e. robots within 2m distance) to which messages can be sent
        """
        return self.neighbors

    def compute_controller(self, type="s"):
        """
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED

        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """

        # here we implement an example for a consensus algorithm
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()

        # send message of positions to all neighbors indicating our position
        for n in neig:
            self.send_message(n, pos)

        # check if we received the position of our neighbors and compute desired change in position
        # as a function of the neighbors (message is composed of [neighbors id, position])
        dx = 0.
        dy = 0.
        if messages:
            # messages = [msg1, msg2, msg3]
            # msg = [id, array([x, y, z]) ]

            # Square Formation
            if type == "s":
                dx, dy = self.formation(messages, pos, type="square")
            elif type == "c":
                dx, dy = self.formation(messages, pos, type="circle")

            # compute velocity change for the wheels
            vel_norm = np.linalg.norm([dx, dy])  # norm of desired velocity
            if vel_norm < 0.01:
                vel_norm = 0.01
            des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
            right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            self.set_wheel_velocity([left_wheel, right_wheel])

        return dx, dy

    def formation(self, msgs, r_pos, type="square"):
        dx = 0
        dy = 0
        # Square Formation
        if type == "square":
            des_coord = np.array([[0, -0.5], [0.5, -0.5 + 1/4], [0, 0.5],
                                  [1, -0.5], [0.5, 0.5 - 1/4], [1, 0.5]])

        # Circle Formation
        if type == "circle":
            des_coord = np.array([[-0.5, -math.sqrt(3)/2], [-1, 0], [-0.5, math.sqrt(3)/2],
                                  [0.5, -math.sqrt(3)/2], [1, 0], [0.5, math.sqrt(3)/2]])

        for msg in msgs:
            n_id = msg[0]
            n_pos = msg[1]

            des_x, des_y = des_coord[self.id] - des_coord[n_id]
            l_x, l_y, _ = r_pos - n_pos

            # Collision Avoidance
            if math.sqrt(math.pow(l_x, 2) + math.pow(l_y, 2)) <= 0.4:
                K = 100
                return K*l_x, K*l_y

            x_r, y_r, _ = r_pos
            x_n, y_n, _ = n_pos
            dx = dx - self.square_formation_control(des_x, l_x, x_r, x_n)
            dy = dy - self.square_formation_control(des_y, l_y, y_r, y_n)

        lim = 50
        # Clip velocity
        dx = np.clip(dx, -lim, lim)
        dy = np.clip(dy, -lim, lim)

        return dx, dy

    def square_formation_control(self, d, l, x_r, x_n):
        delta = 2
        nmrtr = 2*(delta - math.fabs(d)) - math.fabs(l-d)
        dnmntr = delta - math.fabs(d) - math.fabs(l-d)
        dnmntr = math.pow(dnmntr, 2)
        alpha = x_r - x_n - d

        return (nmrtr/dnmntr) * alpha
