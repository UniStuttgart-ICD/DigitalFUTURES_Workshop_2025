'''
The file is set up using the example here: 
https://compas-rrc.github.io/compas_rrc/latest/reference/generated/compas_rrc.AbbClient.html#compas_rrc.AbbClient

'''

import compas_rrc as rrc
import time
from RobotInterface import RobotInterface


class ABBGoFa(RobotInterface):
    def __init__(self, abb_name, client, auto_home = False):
        print(f"Intialising {abb_name} ...")
        super().__init__(abb_name, client)

        self.ros = rrc.RosClient("127.0.0.1")
        self.ros.run()
        print(f"ROS connected: {self.ros.is_connected}")
        
        self.abb = rrc.AbbClient(self.ros, f"/{self.name}")
        print("ABB client connected.")

        # done = self.abb.send_and_wait(rrc.PrintText(f"Hello, {self.name}"))
        # abb.send(rrc.SetTool("tool0"))
        # abb.send(rrc.SetWorkObject("wobj0"))

        super().init_home(auto_home)
    
    def get_current_state(self):
        frame = self.abb.send_and_wait(rrc.GetFrame())
        print(f"{self.name} - current frame: {frame}")

    def go_to_home(self):
        joints, ea = self.abb.send_and_wait(rrc.GetJoints())
        speed = 100
        joints.rax_1 = 0
        joints.rax_2 = 0
        joints.rax_3 = 0
        joints.rax_4 = 0
        joints.rax_5 = 0
        joints.rax_6 = 0
        zone = rrc.Zone.FINE
        move_instr = rrc.MoveToJoints(joints, [], speed, zone)
        self.abb.send_and_wait(move_instr)
    
    def execute(self, frames):
        speed = 100 # mm/s
        zone = rrc.Zone.Z10 
        '''
        Z10 = Movement terminates as a fly-by point (10mm) // program execution continues about 100 ms before robot reaches the zone
        See: https://compas-rrc.github.io/compas_rrc/latest/api/generated/compas_rrc.Zone.html
        
        zone = rrc.Zone.Z5 # (zone: 5 mm) 
        zone = rrc.Zone.Z0 # (zone: 0.3 mm) 
        zone = rrc.Zone.FINE # the movement terminates as a stop point // program execution will not continue until robot reaches stop point
        '''
        for frame in frames:
            if self.stop:
                print ("stopping before completing the trajectory")
                break
            if self.pause:
                print ("pausing")
                continue
            joint_values = frame['positions']
            move_instr = rrc.MoveToJoints(joint_values, [], speed, zone)
            self.abb.send(move_instr)
        if self.stop:
            self.stop = False
            print ("trajectory execution aborted")
        else:
            print ("trajectory execution complete")

    def cleanup(self):
        super().cleanup_connections()
        self.abb.terminate()
        self.ros.terminate()
        



if __name__ == '__main__':
    abb = ABBGoFa("GoFa1", '127.0.0.1')
    
    try:
        while True:
            time.sleep(2)
            # print ("listening for trajectories ...")
    except KeyboardInterrupt:
        abb.cleanup()
