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
static float m = CF_MASS;
static float massThrust = 132000;                                 // emperical value for hovering.

// PID parameters
static float kp_wxy = 2000.0;
static float kd_wxy = 20.0;
static float ki_wxy = 500.0;
static float i_range_wxy = 1.0;

static float kp_wz = 1200.0;
static float kd_wz = 12.0;
static float ki_wz = 1000;
static float i_range_wz = 1.0;

static float kp_accz = 1.0;
static float ki_accz = 0.1;
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
  float wx = radians(sensors->gyro.x);
  float wy = -radians(sensors->gyro.y);
  float wz = radians(sensors->gyro.z);
  // NOTE: acc_z's unit is Gs, need to convert to m/s^2
  // NOTE: need to check az frame
  float az = state->acc.z * 9.81f;

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

  d_error_wx = (p_error_wx - prev_p_error_wx) / dt;
  d_error_wy = (p_error_wy - prev_p_error_wy) / dt;
  d_error_wz = (p_error_wz - prev_p_error_wz) / dt;

  prev_p_error_wx = p_error_wx;
  prev_p_error_wy = p_error_wy;
  prev_p_error_wz = p_error_wz;

  float alphax_des = kp_wxy * p_error_wx + kd_wxy * d_error_wx + ki_wxy * i_error_wx;
  float alphay_des = kp_wxy * p_error_wy + kd_wxy * d_error_wy + ki_wxy * i_error_wy;
  float alphaz_des = kp_wz * p_error_wz + kd_wz * d_error_wz + ki_wz * i_error_wz;
  float az_thrust_des = kp_accz * p_error_accz + ki_accz * i_error_accz;

  // convert into torque and thrust
  struct vec I = {16.571710e-6, 16.571710e-6, 29.261652e-6}; // moment of inertia
  // torque = I * alpha + w x I * w
  torquex_des = alphax_des * I.x + wx * (wy * I.z - wz * I.y);
  torquey_des = alphay_des * I.y + wy * (wz * I.x - wx * I.z);
  torquez_des = alphaz_des * I.z + wz * (wx * I.y - wy * I.x);
  thrust_des = m * az_thrust_des;

  // Sending values to the motor
  control->torqueX = torquex_des;
  control->torqueY = torquey_des;
  control->torqueZ = torquez_des;
  control->thrust = thrust_des;
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