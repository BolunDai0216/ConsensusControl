import numpy as np
import pybullet as p
import itertools
import keyboard

from robot import Robot


class World():
    def __init__(self):
        # create the physics simulator
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)

        self.max_communication_distance = 5.0

        # We will integrate every 4ms (250Hz update)
        self.dt = 1./250.
        p.setPhysicsEngineParameter(self.dt, numSubSteps=1)

        # Create the plane.
        self.planeId = p.loadURDF("../models/plane.urdf")
        p.changeDynamics(self.planeId, -1, lateralFriction=5., rollingFriction=0)

        self.goalId = p.loadURDF("../models/goal.urdf")  # red
        self.goalId = p.loadURDF("../models/goal2.urdf")  # purple

        # the balls
        self.ball1 = p.loadURDF("../models/ball1.urdf")  # purple
        p.resetBasePositionAndOrientation(self.ball1, [1., 4., 0.5], (0., 0., 0.5, 0.5))
        self.ball2 = p.loadURDF("../models/ball2.urdf")  # red
        p.resetBasePositionAndOrientation(self.ball2, [4., 2., 0.5], (0., 0., 0.5, 0.5))

        p.resetDebugVisualizerCamera(7.0, 90.0, -43.0, (1., 1., 0.0))

        # Add objects
        wallId = p.loadSDF("../models/walls.sdf")[0]
        p.resetBasePositionAndOrientation(wallId, [0., -1., 0], (0., 0., 0.5, 0.5))
        wallId = p.loadSDF("../models/walls.sdf")[0]
        p.resetBasePositionAndOrientation(wallId, [0., 1., 0], (0., 0., 0.5, 0.5))
        wallId = p.loadSDF("../models/walls.sdf")[0]
        p.resetBasePositionAndOrientation(wallId, [3., -1., 0], (0., 0., 0.5, 0.5))
        wallId = p.loadSDF("../models/walls.sdf")[0]
        p.resetBasePositionAndOrientation(wallId, [3., 1., 0], (0., 0., 0.5, 0.5))
        wallId = p.loadSDF("../models/walls.sdf")[0]
        p.resetBasePositionAndOrientation(wallId, [1., 2., 0], (0., 0., 0., 1.))
        wallId = p.loadSDF("../models/walls.sdf")[0]
        p.resetBasePositionAndOrientation(wallId, [2., -2., 0], (0., 0., 0., 1.))

        # define initial configuration
        init_dis = 1
        init_bias = 1
        row = 2
        column = 3

        # create 6 robots
        self.robots = []
        for (i, j) in itertools.product(range(row), range(column)):
            self.robots.append(
                Robot([init_dis*i+init_bias, init_dis*j-init_bias, 0.3], column*i+j, self.dt))
            # self.robots.append(
            #     Robot([2, -1, 0.3], column*i+j, self.dt))

        self.time = 0.0
        self.type = "s"

        self.stepSimulation()
        self.stepSimulation()

    def reset(self):
        """
        Resets the position of all the robots
        """
        for r in self.robots:
            r.reset()
        p.stepSimulation()

    def stepSimulation(self):
        """
        Simulates one step simulation
        """

        # for each robot construct list of neighbors
        for r in self.robots:
            r.neighbors = []  # reset neighbors
            r.messages_received = []  # reset message received
            pos1, or1 = r.get_pos_and_orientation()
            for j, r2 in enumerate(self.robots):
                if(r.id != r2.id):
                    pos2, or2 = r2.get_pos_and_orientation()
                    if(np.linalg.norm(pos1-pos2) < self.max_communication_distance):
                        r.neighbors.append(j)

        # for each robot send and receive messages
        for i, r in enumerate(self.robots):
            for msg in r.messages_to_send:
                if msg[0] in r.neighbors:  # then we can send the message
                    self.robots[msg[0]].messages_received.append([i, msg[1]])  # add the sender id
            r.messages_to_send = []

        # update the controllers
        if self.time > 1.0:
            for r in self.robots:
                if keyboard.is_pressed('c'):
                    print('Circular Formation')
                    self.type = "c"
                elif keyboard.is_pressed('s'):
                    print('Square Formation')
                    self.type = "s"
                elif keyboard.is_pressed('l'):
                    print('Virtual Leader')
                    self.type = "l"
                elif keyboard.is_pressed('e'):
                    print('Virtual Leader 2')
                    self.type = "e"
                elif keyboard.is_pressed('p'):
                    print('Purple')
                    self.type = "p"
                elif keyboard.is_pressed('d'):
                    print('Push Purple')
                    self.type = "d"
                elif keyboard.is_pressed('r'):
                    print('Red')
                    self.type = "r"
                elif keyboard.is_pressed('b'):
                    print('Big Circle')
                    self.type = "b"
                elif keyboard.is_pressed('t'):
                    print('Push Red')
                    self.type = "t"
                elif keyboard.is_pressed('o'):
                    print('Big circle')
                    self.type = "o"
                elif keyboard.is_pressed('d'):
                    print('Diamond')
                    self.type = "d"
                dx, dy = r.compute_controller(self.type)

        # do one simulation step
        p.stepSimulation()
        self.time += self.dt
