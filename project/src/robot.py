import numpy as np
import pybullet as p
import controller


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

        self.purple_x = 1
        self.purple_y = 4
        self.red_x = 4
        self.red_y = 2

        self.init_push = True
        self.des_xs = [0, 0, 0, 0, 0, 0, 0]
        self.des_ys = [0, 0, 0, 0, 0, 0, 0]
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

    def compute_controller(self, time):
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

            if time < 5:
                # square Formation
                dx, dy = controller.formation_control("square", messages, self.id, pos)

            elif time >= 5 and time < 30:
                # move out of room
                dx, dy = controller.follow_leader(3, 4, pos, messages,
                                                  leader_alpha=30)

            elif time >= 30 and time < 65:
                # circle formation
                dx, dy = controller.formation_control("circle", messages, self.id, pos)

            elif time >= 65 and time < 75:
                # surround purple ball
                dx, dy = controller.go_to_leader(self.purple_x, self.purple_y, pos,
                                                 messages, 0, 0)
                self.init_push = True

            elif time >= 75 and time < 115:
                # push purple ball
                if self.init_push:
                    output = controller.get_current_formation(messages, self.id, pos)
                    self.des_xs, self.des_ys, self.cx, self.cy = output
                    self.init_push = False

                dx, dy, self.cx, self.cy = controller.push_ball(
                    messages, self.id, pos, self.des_xs, self.des_ys,
                    self.cx, self.cy, 2.5, 5.5, K=3)

            elif time >= 115 and time < 130:
                # eject from purple ball
                dx, dy = controller.eject_from_ball(2.5, 5.5, messages, pos, d_0=1.5)

            elif time >= 130 and time < 155:
                # surround red ball
                dx, dy = controller.go_to_leader(self.red_x, self.red_y, pos,
                                                 messages,
                                                 self.purple_x, self.purple_y,
                                                 obstacle=True, room_alpha=1)
                self.init_push = True

            elif time >= 155 and time < 205:
                # push red ball
                if self.init_push:
                    output = controller.get_current_formation(messages, self.id, pos)
                    self.des_xs, self.des_ys, self.cx, self.cy = output
                    self.init_push = False

                dx, dy, self.cx, self.cy = controller.push_ball(
                    messages, self.id, pos, self.des_xs, self.des_ys,
                    self.cx, self.cy, 0.5, 5.5, K=3)

            elif time >= 205 and time < 220:
                # eject from red ball
                dx, dy = controller.eject_from_ball(0.5, 5.5, messages, pos)

            elif time >= 220 and time < 230:
                # return to room step 1
                dx, dy = controller.follow_leader(2, 3.5, pos, messages)

            elif time >= 230 and time < 270:
                # return to room step 2
                dx, dy = controller.follow_leader(2, 0, pos, messages,
                                                  leader_alpha=30, agent_alpha=3)
            elif time >= 270 and time < 290:
                # return to room step 3
                dx, dy = controller.follow_leader(1.5, 0, pos, messages)

            elif time >= 290:
                # diamond formation
                dx, dy = controller.formation_control("diamond", messages, self.id, pos)

            # compute velocity change for the wheels
            vel_norm = np.linalg.norm([dx, dy])  # norm of desired velocity
            if vel_norm < 0.01:
                vel_norm = 0.01
            des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
            right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            self.set_wheel_velocity([left_wheel, right_wheel])

        return dx, dy
