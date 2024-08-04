import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class CF2Sim(Node):
    def __init__(self):
        super().__init__("mujoco_env")
        # control related
        self.ctrl_dt = 0.02
        self.real_time_factor = 1.0
        self.sim_dt = 0.005
        self.n_acts = 100
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.t = 0.0
        # mujoco setup
        self.mj_model = mujoco.MjModel.from_xml_path(
            f"{os.path.dirname(os.path.abspath(__file__))}/model/scene.xml"
        )
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.acts = np.zeros([self.n_acts, self.mj_model.nu])
        self.ctrl_hover = np.ones(4) * 0.06622
        self.plan_time = 0.0
        # ros setup
        self.state_pub = self.create_publisher(Float32MultiArray, "state", 10)
        self.act_sub = self.create_subscription(
            Float32MultiArray, "action", self.act_callback, 20
        )

    def act_callback(self, msg):
        data = np.array(msg.data)
        self.plan_time = data[0]
        thrust_des = data[1:].reshape(-1, 4)
        self.n_acts = thrust_des.shape[0]
        self.acts[: self.n_acts] = thrust_des
        self.acts[self.n_acts :] = self.ctrl_hover

    def main_loop(self):
        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while rclpy.ok():
                t0 = time.time()
                rclpy.spin_once(self, timeout_sec=0.001)
                delta_time = self.t - self.plan_time
                delta_step = int(delta_time / self.ctrl_dt)
                if delta_time > 0.02:
                    self.get_logger().warn(
                        f"Delayed by {delta_time*1000.0:.1f} ms"
                    )
                if delta_step >= self.n_acts or delta_step < 0:
                    # self.get_logger().warn(
                    #     f"Control signal outdated by {delta_time*1000.0:.1f} ms"
                    # )
                    delta_step = self.n_acts - 1
                
                # for _ in range(self.n_frame):
                self.mj_data.ctrl = self.acts[delta_step]
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt
                q = self.mj_data.qpos 
                qd = self.mj_data.qvel 
                state = np.concatenate([[self.t], q, qd])
                self.state_pub.publish(Float32MultiArray(data=state))

                viewer.sync()
                t1 = time.time()
                if t1 - t0 < self.sim_dt:
                    time.sleep((self.sim_dt - (t1 - t0)) / self.real_time_factor)
                else:
                    self.get_logger().warn("Control loop overruns")


def main(args=None):
    rclpy.init(args=args)

    mujoco_env = CF2Sim()

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        pass

    mujoco_env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()