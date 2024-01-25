#include <math.h>

#include "param.h"
#include "log.h"
#include "math3d.h"
#include "position_controller.h"
#include "controller_indi.h"
#include "physicalConstants.h"

/*
NOTEs: the unit use here
1. angle: radian
2. acceleration: m/s^2
*/

// Dynamic parameters
static float g_vehicleMass = CF_MASS;
static float massThrust = 132000; // emperical value for hovering. 

// PID parameters
static float kp_wxy = 2000.0;
static float kd_wxy = 20.0;
static float ki_wxy = 500.0;
static float i_range_wxy = 1.0;

static float kp_wz = 1200.0;
static float kd_wz = 12.0;
static float ki_wz = 1000;
static float i_range_wz = 1.0;

// PID intermediate variables
static float p_error_wx = 0.0;
static float p_error_wy = 0.0;
static float p_error_wz = 0.0;
static float i_error_wx = 0.0;
static float i_error_wy = 0.0;
static float i_error_wz = 0.0;
static float d_error_wx = 0.0;
static float d_error_wy = 0.0;
static float d_error_wz = 0.0;

// Setpoint variables
static float omega_x;
static float omega_y;
static float omega_z;
static float acc_z;
static float omega_x_des;
static float omega_y_des;
static float omega_z_des;
static float acc_z_des;

void controllerINDIReset(void)
{
  i_error_wx = 0;
  i_error_wy = 0;
  i_error_wz = 0;
}

void controllerINDIInit(void)
{
  controllerINDIReset();
}

bool controllerINDITest(void)
{
  return true;
}

void controllerINDI(control_t *control, const setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
{
  // set to custom power distribution controller
  control->controlMode = controlModeForce;

  struct vec ew, M;
  float dt;
  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, tick)) {
    return;
  }

  dt = (float)(1.0f/ATTITUDE_RATE);

  // Angular velocity setpoint
  omega_x = radians(setpoint->attitudeRate.roll);
  omega_y = -radians(setpoint->attitudeRate.pitch);
  omega_z = radians(setpoint->attitudeRate.yaw);

  // Angular velocity Controller
  float stateAttitudeRateRoll = radians(sensors->gyro.x);
  float stateAttitudeRatePitch = -radians(sensors->gyro.y);
  float stateAttitudeRateYaw = radians(sensors->gyro.z);

  p_error_wx = omega_x - stateAttitudeRateRoll;
  p_error_wy = omega_y - stateAttitudeRatePitch;
  p_error_wz = omega_z - stateAttitudeRateYaw;

  i_error_wx += p_error_wx * dt;
  i_error_wx = clamp(i_error_wx, -i_range_wxy, i_range_wxy);
  i_error_wy += p_error_wy * dt;
  i_error_wy = clamp(i_error_wy, -i_range_wxy, i_range_wxy);
  i_error_wz += p_error_wz * dt;
  i_error_wz = clamp(i_error_wz, -i_range_wz, i_range_wz);

  d_error_wx = (p_error_wx - prev_p_error_wx) / dt;

  // for logging
  i_roll = i_error_omega_roll;
  i_pitch = i_error_omega_pitch;
  i_yaw = i_error_omega_yaw;

  // derivative terms
  float err_d_roll = 0;
  float err_d_pitch = 0;
  if (prev_omega_roll == prev_omega_roll) { /*d part initialized*/
    err_d_roll = ((radians(setpoint->attitudeRate.roll) - prev_setpoint_omega_roll) - (stateAttitudeRateRoll - prev_omega_roll)) / dt;
    err_d_pitch = (-(radians(setpoint->attitudeRate.pitch) - prev_setpoint_omega_pitch) - (stateAttitudeRatePitch - prev_omega_pitch)) / dt;
  }
  prev_omega_roll = stateAttitudeRateRoll;
  prev_omega_pitch = stateAttitudeRatePitch;
  prev_setpoint_omega_roll = radians(setpoint->attitudeRate.roll);
  prev_setpoint_omega_pitch = radians(setpoint->attitudeRate.pitch);

  M.x = kw_xy * ew.x + ki_w_x*i_error_omega_roll + kd_omega_rp*err_d_roll; 
  M.y = kw_xy * ew.y + ki_w_y*i_error_omega_pitch + kd_omega_rp*err_d_pitch;
  M.z = kw_z  * ew.z + ki_w_z*i_error_omega_yaw;

  // Sending values to the motor
  control->thrust = massThrust * g_vehicleMass * setpoint->acceleration.z;
  thrust_des = control->thrust;
  r_roll = radians(sensors->gyro.x);
  r_pitch = -radians(sensors->gyro.y);
  r_yaw = radians(sensors->gyro.z);
  acc_z = sensors->acc.z;

  if (control->thrust > 0) {
    control->roll = clamp(M.x, -32000, 32000);
    control->pitch = clamp(M.y, -32000, 32000);
    control->yaw = clamp(-M.z, -32000, 32000);

    omega_x_des = control->roll;
    omega_y_des = control->pitch;
    omega_z_des = control->yaw;

  } else {
    control->roll = 0;
    control->pitch = 0;
    control->yaw = 0;

    omega_x_des = control->roll;
    omega_y_des = control->pitch;
    omega_z_des = control->yaw;

    controllerINDIReset();
  }
}

PARAM_GROUP_START(ctrlRwik)
PARAM_ADD(PARAM_FLOAT, mass, &g_vehicleMass)
PARAM_ADD(PARAM_FLOAT, massThrust, &massThrust)
PARAM_ADD(PARAM_FLOAT, kw_xy, &kw_xy)
PARAM_ADD(PARAM_FLOAT, kw_z, &kw_z)
PARAM_ADD(PARAM_FLOAT, kd_omega_rp, &kd_omega_rp)
PARAM_ADD(PARAM_FLOAT, ki_w_z, &ki_w_z)
PARAM_ADD(PARAM_FLOAT, ki_w_x, &ki_w_x)
PARAM_ADD(PARAM_FLOAT, ki_w_y, &ki_w_y)

PARAM_GROUP_STOP(ctrlRwik)

LOG_GROUP_START(ctrlRwik)
LOG_ADD(LOG_FLOAT, acc_z_des, &thrust_des)
LOG_ADD(LOG_FLOAT, omega_x_des, &omega_x_des)
LOG_ADD(LOG_FLOAT, omega_y_des, &omega_y_des)
LOG_ADD(LOG_FLOAT, omega_z_des, &omega_z_des)
LOG_ADD(LOG_FLOAT, acc_z, &acc_z)

LOG_ADD(LOG_FLOAT, omega_x, &omega_x)
LOG_ADD(LOG_FLOAT, omega_y, &omega_y)
LOG_ADD(LOG_FLOAT, omega_z, &omega_z)

LOG_ADD(LOG_FLOAT, cmd_z_acc, &cmd_z_acc)

LOG_GROUP_STOP(ctrlRwik)