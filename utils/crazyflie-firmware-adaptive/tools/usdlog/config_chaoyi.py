1     # version
8192   # buffer size in bytes
log   # file name
0     # enable on startup (0/1)
on:fixedFrequency
100     # frequency
1     # mode (0: disabled, 1: synchronous stabilizer, 2: asynchronous)

# esitimated value

# stateEstimate.x
# stateEstimate.y
# stateEstimate.z

# stateEstimate.vx
# stateEstimate.vy
# stateEstimate.vz

# stateEstimate.ax
# stateEstimate.ay
# stateEstimate.az

# stateEstimate.roll
# stateEstimate.pitch
# stateEstimate.yaw

gyro.x
gyro.y
gyro.z

# target value

# ctrltarget.x
# ctrltarget.y
# ctrltarget.z

# ctrltarget.vx
# ctrltarget.vy
# ctrltarget.vz

# ctrltarget.ax
# ctrltarget.ay
# ctrltarget.az

ctrltarget.roll
ctrltarget.pitch
ctrltarget.yaw

# controller desired values

# input command
# controller.cmd_thrust
controller.cmd_roll
controller.cmd_pitch
controller.cmd_yaw
# desired value
controller.roll 
controller.pitch
controller.yaw
controller.rollRate
controller.pitchRate
controller.yawRate

# controller observed value
controller.r_roll
controller.r_pitch
controller.r_yaw