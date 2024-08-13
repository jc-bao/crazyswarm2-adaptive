import logging
import random
import time
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

address = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class CF2Real:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """Initialize and run the example with the specified link_uri"""

        self._cf = Crazyflie(rw_cache="./cache")

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.fully_connected.add_callback(self._fully_connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print("Connecting to %s" % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        self._param_check_list = []
        self._param_groups = []

        random.seed()

        self.t = time.time()

    def _connected(self, link_uri):
        print("Connected to %s" % link_uri)

    def _fully_connected(self, link_uri):
        """This callback is called when the Crazyflie has been connected and all parameters have been
        downloaded. It is now OK to set and get parameters."""
        # print(f"Parameters downloaded to {link_uri}")

        # We can get a parameter value directly without using a callback
        # value = self._cf.param.get_value("pid_attitude.pitch_kd")
        # print(f"Value read with get() is {value}")

        # When a parameter is set, the callback is called with the new value
        # self._cf.param.add_update_callback(
        #     group="pid_attitude", name="pitch_kd", cb=self._a_pitch_kd_callback
        # )
        # When setting a value the parameter is automatically read back
        # and the registered callbacks will get the updated value
        # self._cf.param.set_value("pid_attitude.pitch_kd", 0.1234)

        print(f"freq={1/(time.time()-self.t):.1f}")
        self.t = time.time()

        self._cf.param.set_value("motorPowerSet.enable", 1)
        pwm = 0.0
        for i in range(1,5):
            self._cf.param.set_value(f"motorPowerSet.m{i}", pwm)


    def _cpu_flash_callback(self, name, value):
        """Specific callback for the cpu.flash parameter"""
        print("The connected Crazyflie has {}kb of flash".format(value))

    def _param_callback(self, name, value):
        """Generic callback registered for all the groups"""
        print("{0}: {1}".format(name, value))

        # Remove each parameter from the list when fetched
        self._param_check_list.remove(name)
        if len(self._param_check_list) == 0:
            print("Have fetched all parameter values.")

            # Remove all the group callbacks
            for g in self._param_groups:
                self._cf.param.remove_update_callback(group=g, cb=self._param_callback)

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print("Connection to %s failed: %s" % (link_uri, msg))
        self.is_connected = False
        self._cf.param.set_value("motorPowerSet.enable", 0)

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print("Connection to %s lost: %s" % (link_uri, msg))
        self._cf.param.set_value("motorPowerSet.enable", 0)

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print("Disconnected from %s" % link_uri)
        self.is_connected = False
        self._cf.param.set_value("motorPowerSet.enable", 0)

    def main_loop(self):
        t = time.time()
        pwm = 10000 * (np.sin(2 * np.pi * 3.0 * t)+1.0)
        for i in range(1,5):
            self._cf.param.set_value(f"motorPowerSet.m{i}", pwm)


if __name__ == "__main__":
    import logging

    logging.getLogger().setLevel(logging.INFO)
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    pe = CF2Real(address)

    # The Crazyflie lib doesn't contain anything to keep the application
    # alive, so this is where your application should do something. In our
    # case we are just waiting until we are disconnected.
    while pe.is_connected:
        pe.main_loop()
        time.sleep(0.01)
