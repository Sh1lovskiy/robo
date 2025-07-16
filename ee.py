from robot.controller import RobotController
import numpy as np

controller = RobotController()
controller.enable()
controller.connect()
print(controller.get_tcp_pose())
pose = [-120.0, -220.0, 250.0, 180.0, 0.0, 0.0]
if not controller.check_safety():
    exit()
success = controller.move_linear(pose)
if success:
    controller.wait_motion_done()
    print("TCP after:", controller.get_tcp_pose())
print("Success:", success)
print("TCP after:", controller.get_tcp_pose())
