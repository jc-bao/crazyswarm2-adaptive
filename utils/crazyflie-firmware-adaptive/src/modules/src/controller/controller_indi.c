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
static float m = 0.0411;
static float massThrust = 85000;                                 // emperical value for hovering.

// PID parameters
static float kp_wxy = 750.0;
static float kd_wxy = 8.0;
static float ki_wxy = 500.0;
static float i_range_wxy = 1.0;

static float kp_wz = 400.0;
static float kd_wz = 2.0;
static float ki_wz = 200.0;
static float i_range_wz = 1.0;

static float kp_accz = 1.0;
static float ki_accz = 0.2;
static float i_range_accz = 0.2;

// PID intermediate variables
static float p_error_wx = 0.0;
static float p_error_wy = 0.0;
static float p_error_wz = 0.0;
static float p_error_accz = 0.0;
static float i_error_wx = 0.0;
static float i_error_wy = 0.0;
static float i_error_wz = 0.0;
static float i_error_accz = 0.0;
static float d_error_wx = 0.0;
static float d_error_wy = 0.0;
static float d_error_wz = 0.0;
static float prev_p_error_wx = 0.0;
static float prev_p_error_wy = 0.0;
static float prev_p_error_wz = 0.0;
static float prev_d_error_wx = 0.0;
static float prev_d_error_wy = 0.0;
static float prev_d_error_wz = 0.0;

// PID output
static float torquex_des = 0.0;
static float torquey_des = 0.0;
static float torquez_des = 0.0;
static float thrust_des = 0.0;

// Setpoint variables
static float wx;
static float wy;
static float wz;
static float az;
static float wx_des;
static float wy_des;
static float wz_des;
static float az_des;

void controllerINDIReset(void)
{
  i_error_wx = 0;
  i_error_wy = 0;
  i_error_wz = 0;
  i_error_accz = 0;
  prev_p_error_wx = 0;
  prev_p_error_wy = 0;
  prev_p_error_wz = 0;
  prev_d_error_wx = 0;
  prev_d_error_wy = 0;
  prev_d_error_wz = 0;
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
//   set to custom power distribution controller
//   control->controlMode = controlModeForce;
//   control->controlMode = controlModeForceTorque;
  control->controlMode = controlModeLegacy;

  float dt;
  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, tick))
  {
    return;
  }
  dt = (float)(1.0f / ATTITUDE_RATE);

  // Angular velocity setpoint
  wx_des = radians(setpoint->attitudeRate.roll);
  wy_des = -radians(setpoint->attitudeRate.pitch);
  wz_des = radians(setpoint->attitudeRate.yaw);
  az_des = setpoint->acceleration.z;
  // NOTE: gyro observation might be noisy, need to check
  wx = radians(sensors->gyro.x);
  wy = -radians(sensors->gyro.y);
  wz = radians(sensors->gyro.z);
  // NOTE: acc_z's unit is Gs, need to convert to m/s^2, and acc_z does not include gravity
  // NOTE: need to check az frame
  az = (sensors->acc.z) * 9.81f;

  // PID controller
  p_error_wx = wx_des - wx;
  p_error_wy = wy_des - wy;
  p_error_wz = wz_des - wz;
  p_error_accz = az_des - az;

  i_error_wx += p_error_wx * dt;
  i_error_wx = clamp(i_error_wx, -i_range_wxy, i_range_wxy);
  i_error_wy += p_error_wy * dt;
  i_error_wy = clamp(i_error_wy, -i_range_wxy, i_range_wxy);
  i_error_wz += p_error_wz * dt;
  i_error_wz = clamp(i_error_wz, -i_range_wz, i_range_wz);
  i_error_accz += p_error_accz * dt;
  i_error_accz = clamp(i_error_accz, -i_range_accz, i_range_accz);

  float new_d_error_wx = (p_error_wx - prev_p_error_wx) / dt;
  float new_d_error_wy = (p_error_wy - prev_p_error_wy) / dt;
  float new_d_error_wz = (p_error_wz - prev_p_error_wz) / dt;

  // go through low pass filter
  float alpha = 1.0f;
  d_error_wx = alpha * new_d_error_wx + (1 - alpha) * prev_d_error_wx;
  d_error_wy = alpha * new_d_error_wy + (1 - alpha) * prev_d_error_wy;
  d_error_wz = alpha * new_d_error_wz + (1 - alpha) * prev_d_error_wz;

  prev_p_error_wx = p_error_wx;
  prev_p_error_wy = p_error_wy;
  prev_p_error_wz = p_error_wz;
  prev_d_error_wx = d_error_wx;
  prev_d_error_wy = d_error_wy;
  prev_d_error_wz = d_error_wz;

  float alphax_des = kp_wxy * p_error_wx + kd_wxy * d_error_wx + ki_wxy * i_error_wx;
  float alphay_des = kp_wxy * p_error_wy + kd_wxy * d_error_wy + ki_wxy * i_error_wy;
  float alphaz_des = kp_wz * p_error_wz + kd_wz * d_error_wz + ki_wz * i_error_wz;
  float az_thrust_des = kp_accz * p_error_accz + ki_accz * i_error_accz + 9.81f;

  // convert into torque and thrust
  struct vec I = {16.571710e-6, 16.571710e-6, 29.261652e-6}; // moment of inertia
  // single motor thrust limit = 0.19N, max torque = 1e-2 N.m
  // torque = I * alpha + w x I * w
  torquex_des = alphax_des * I.x + wx * (wy * I.z - wz * I.y);
  torquey_des = alphay_des * I.y + wy * (wz * I.x - wx * I.z);
  torquez_des = alphaz_des * I.z + wz * (wx * I.y - wy * I.x);
  thrust_des = m * az_thrust_des;
  // torquex_des = clamp(torquex_des, -1e-2, 1e-2);
  // torquey_des = clamp(torquey_des, -1e-2, 1e-2);
  // torquez_des = clamp(torquez_des, -1e-2, 1e-2);
  // thrust_des = clamp(thrust_des, 0.0, 0.19);

  // Sending values to the motor
  float arm = 0.046f * 0.707f;
  float torquex_pwm = 0.25f / arm * torquex_des * 5.188f * 65535.0f;
  float torquey_pwm = 0.25f / arm * torquey_des * 5.188f * 65535.0f;
  float torquez_pwm = 0.25f * torquez_des / 0.005964552f * 65535.0f;
  float thrust_pwm = 0.25f * thrust_des * 5.188f * 65535.0f;
  // float thrust_pwm = 0.041f * (setpoint->acceleration.z) * massThrust; 
  control->roll = clamp(torquex_pwm, -32000, 32000);
  control->pitch = clamp(torquey_pwm, -32000, 32000);
  control->yaw = clamp(-torquez_pwm, -32000, 32000);
  if (setpoint->mode.z == modeDisable) {
    control->thrust = setpoint->thrust;
  } else {
    control->thrust = thrust_pwm;
  }
  if (control->thrust < 0) {
    control->thrust = 0;
    control->roll = 0;
    control->pitch = 0;
    control->yaw = 0;
    controllerINDIReset();
  }
  // control->tau_x = torquex_des;
  // control->tau_y = torquey_des;
  // control->tau_z = torquez_des;
  // if (setpoint->mode.z == modeDisable) {
  //   control->T = setpoint->thrust;
  // } else {
  //   control->T = thrust_des;
  // }
}

PARAM_GROUP_START(ctrlRwik)
PARAM_ADD(PARAM_FLOAT, m, &m)
PARAM_ADD(PARAM_FLOAT, massThrust, &massThrust)
PARAM_ADD(PARAM_FLOAT, kp_wxy, &kp_wxy)
PARAM_ADD(PARAM_FLOAT, kd_wxy, &kd_wxy)
PARAM_ADD(PARAM_FLOAT, ki_wxy, &ki_wxy)
PARAM_ADD(PARAM_FLOAT, i_range_wxy, &i_range_wxy)
PARAM_ADD(PARAM_FLOAT, kp_wz, &kp_wz)
PARAM_ADD(PARAM_FLOAT, kd_wz, &kd_wz)
PARAM_ADD(PARAM_FLOAT, ki_wz, &ki_wz)
PARAM_ADD(PARAM_FLOAT, i_range_wz, &i_range_wz)
PARAM_ADD(PARAM_FLOAT, kp_accz, &kp_accz)
PARAM_ADD(PARAM_FLOAT, ki_accz, &ki_accz)
PARAM_ADD(PARAM_FLOAT, i_range_accz, &i_range_accz)

PARAM_GROUP_STOP(ctrlRwik)

LOG_GROUP_START(ctrlRwik)
LOG_ADD(LOG_FLOAT, p_error_wx, &p_error_wx)
LOG_ADD(LOG_FLOAT, p_error_wy, &p_error_wy)
LOG_ADD(LOG_FLOAT, p_error_wz, &p_error_wz)
LOG_ADD(LOG_FLOAT, p_error_accz, &p_error_accz)
LOG_ADD(LOG_FLOAT, i_error_wx, &i_error_wx)
LOG_ADD(LOG_FLOAT, i_error_wy, &i_error_wy)
LOG_ADD(LOG_FLOAT, i_error_wz, &i_error_wz)
LOG_ADD(LOG_FLOAT, i_error_accz, &i_error_accz)
LOG_ADD(LOG_FLOAT, d_error_wx, &d_error_wx)
LOG_ADD(LOG_FLOAT, d_error_wy, &d_error_wy)
LOG_ADD(LOG_FLOAT, d_error_wz, &d_error_wz)
LOG_ADD(LOG_FLOAT, prev_p_error_wx, &prev_p_error_wx)
LOG_ADD(LOG_FLOAT, prev_p_error_wy, &prev_p_error_wy)
LOG_ADD(LOG_FLOAT, prev_p_error_wz, &prev_p_error_wz)
LOG_ADD(LOG_FLOAT, torquex_des, &torquex_des)
LOG_ADD(LOG_FLOAT, torquey_des, &torquey_des)
LOG_ADD(LOG_FLOAT, torquez_des, &torquez_des)
LOG_ADD(LOG_FLOAT, thrust_des, &thrust_des)
LOG_ADD(LOG_FLOAT, wx, &wx)
LOG_ADD(LOG_FLOAT, wy, &wy)
LOG_ADD(LOG_FLOAT, wz, &wz)
LOG_ADD(LOG_FLOAT, az, &az)
LOG_ADD(LOG_FLOAT, wx_des, &wx_des)
LOG_ADD(LOG_FLOAT, wy_des, &wy_des)
LOG_ADD(LOG_FLOAT, wz_des, &wz_des)
LOG_ADD(LOG_FLOAT, az_des, &az_des)

LOG_GROUP_STOP(ctrlRwik)