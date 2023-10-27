/*
The MIT License (MIT)

Copyright (c) 2018 Wolfgang Hoenig and James Alan Preiss

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
This controller is based on the following publication:

Daniel Mellinger, Vijay Kumar:
Minimum snap trajectory generation and control for quadrotors.
IEEE International Conference on Robotics and Automation (ICRA), 2011.

We added the following:
 * Integral terms (compensates for: battery voltage drop over time, unbalanced center of mass due to asymmetries, and uneven wear on propellers and motors)
 * D-term for angular velocity
 * Support to use this controller as an attitude-only controller for manual flight
*/

#include <math.h>

#include "param.h"
#include "log.h"
#include "position_controller.h"
#include "controller_mellinger.h"
#include "physicalConstants.h"

#include "debug.h"


#define UPDATE_RATE RATE_500_HZ

static struct mat33 CRAZYFLIE_INERTIA =
    {{{16.6e-6f, 0.83e-6f, 0.72e-6f},
      {0.83e-6f, 16.6e-6f, 1.8e-6f},
      {0.72e-6f, 1.8e-6f, 29.3e-6f}}};

static struct vec OMEGA_GAIN = {1.0f, 1.0f, 1.0f};


// minimum and maximum body rates
static float omega_rp_max = 30;
static float omega_yaw_max = 10;
static float heuristic_rp = 12;
static float heuristic_yaw = 5;

// time constant of rotational rate control
static float tau_rp_rate = 0.015;
static float tau_yaw_rate = 0.0075;

static float dt = 0.002;

// Global state variable used in the
// firmware as the only instance and in bindings
// to hold the default values
static controllerMellinger_t g_self = {
  .mass = CF_MASS,
  .massThrust = 132000,

  // XY Position PID
  .kp_xy = 0.4,       // P
  .kd_xy = 0.2,       // D
  .ki_xy = 0.05,      // I
  .i_range_xy = 2.0,

  // Z Position
  .kp_z = 1.25,       // P
  .kd_z = 0.4,        // D
  .ki_z = 0.05,       // I
  .i_range_z  = 0.4,

  // Attitude
  .kR_xy = 30, // P
  .kw_xy = 20000, // D
  .ki_m_xy = 0.0, // I
  .i_range_m_xy = 1.0,

  // Yaw
  .kR_z = 10, // P
  .kw_z = 12000, // D
  .ki_m_z = 500, // I
  .i_range_m_z  = 1500,

  // roll and pitch angular velocity
  .kd_omega_rp = 200, // D


  // Helper variables
  .i_error_x = 0,
  .i_error_y = 0,
  .i_error_z = 0,

  .i_error_m_x = 0,
  .i_error_m_y = 0,
  .i_error_m_z = 0,
};


void controllerMellingerReset(controllerMellinger_t* self)
{
  self->i_error_x = 0;
  self->i_error_y = 0;
  self->i_error_z = 0;
  self->i_error_m_x = 0;
  self->i_error_m_y = 0;
  self->i_error_m_z = 0;
}

void controllerMellingerInit(controllerMellinger_t* self)
{
  // copy default values (bindings), or does nothing (firmware)
  DEBUG_PRINT("controllerMellingerInit\n");
  *self = g_self;

  controllerMellingerReset(self);
}

bool controllerMellingerTest(controllerMellinger_t* self)
{
  return true;
}

void controllerMellinger(controllerMellinger_t* self, control_t *control, const setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
{
  static float control_omega[3];
  static struct vec control_torque;
  static float control_thrust;

//   DEBUG_PRINT("controllerMellinger\n");

  // define this here, since we do body-rate control at 1000Hz below the following if statement
  float omega[3] = {0};
  omega[0] = radians(sensors->gyro.x);
  omega[1] = radians(sensors->gyro.y);
  omega[2] = radians(sensors->gyro.z);
  if (RATE_DO_EXECUTE(RATE_500_HZ, tick)) {
    // desired accelerations
    // struct vec accDes = vzero();
    // desired thrust
    // float collCmd = 0;

    // // attitude error as computed by the reduced attitude controller
    // struct quat attErrorReduced = qeye();

    // // attitude error as computed by the full attitude controller
    // struct quat attErrorFull = qeye();

    // // desired attitude as computed by the full attitude controller
    // struct quat attDesiredFull = qeye();

    // // current attitude
    // struct quat attitude = mkquat(
    //   state->attitudeQuaternion.x,
    //   state->attitudeQuaternion.y,
    //   state->attitudeQuaternion.z,
    //   state->attitudeQuaternion.w);

    // // inverse of current attitude
    // struct quat attitudeI = qinv(attitude);

    // // body frame -> inertial frame :  vI = R * vB
    // // float R[3][3] = {0};
    // // struct quat q = attitude;
    // // R[0][0] = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;
    // // R[0][1] = 2 * q.x * q.y - 2 * q.w * q.z;
    // // R[0][2] = 2 * q.x * q.z + 2 * q.w * q.y;

    // // R[1][0] = 2 * q.x * q.y + 2 * q.w * q.z;
    // // R[1][1] = q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z;
    // // R[1][2] = 2 * q.y * q.z - 2 * q.w * q.x;

    // // R[2][0] = 2 * q.x * q.z - 2 * q.w * q.y;
    // // R[2][1] = 2 * q.y * q.z + 2 * q.w * q.x;
    // // R[2][2] = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;

    // // We don't need all terms of R, only compute the necessary parts

    // float R02 = 2 * attitude.x * attitude.z + 2 * attitude.w * attitude.y;
    // float R12 = 2 * attitude.y * attitude.z - 2 * attitude.w * attitude.x;
    // float R22 = attitude.w * attitude.w - attitude.x * attitude.x - attitude.y * attitude.y + attitude.z * attitude.z;

    // // a few temporary quaternions
    // struct quat temp1 = qeye();
    // struct quat temp2 = qeye();

  // Chaoyi
  //   // compute the position and velocity errors
  //   struct vec pError = mkvec(setpoint->position.x - state->position.x,
  //                             setpoint->position.y - state->position.y,
  //                             setpoint->position.z - state->position.z);

  //   struct vec vError = mkvec(setpoint->velocity.x - state->velocity.x,
  //                             setpoint->velocity.y - state->velocity.y,
  //                             setpoint->velocity.z - state->velocity.z);


  //   // ====== LINEAR CONTROL ======

  //   // compute desired accelerations in X, Y and Z
  //   accDes.x = 0;
  //   accDes.x += 1.0f / tau_xy / tau_xy * pError.x;
  //   accDes.x += 2.0f * zeta_xy / tau_xy * vError.x;
  //   accDes.x += setpoint->acceleration.x;
  //   accDes.x = constrain(accDes.x, -coll_max, coll_max);

  //   accDes.y = 0;
  //   accDes.y += 1.0f / tau_xy / tau_xy * pError.y;
  //   accDes.y += 2.0f * zeta_xy / tau_xy * vError.y;
  //   accDes.y += setpoint->acceleration.y;
  //   accDes.y = constrain(accDes.y, -coll_max, coll_max);

  //   accDes.z = GRAVITY_MAGNITUDE;
  //   accDes.z += 1.0f / tau_z / tau_z * pError.z;
  //   accDes.z += 2.0f * zeta_z / tau_z * vError.z;
  //   accDes.z += setpoint->acceleration.z;
  //   accDes.z = constrain(accDes.z, -coll_max, coll_max);


  //   // ====== THRUST CONTROL ======

  //   // compute commanded thrust required to achieve the z acceleration
  //   collCmd = accDes.z / R22;

  //   if (fabsf(collCmd) > coll_max) {
  //     // exceeding the thrust threshold
  //     // we compute a reduction factor r based on fairness f \in [0,1] such that:
  //     // collMax^2 = (r*x)^2 + (r*y)^2 + (r*f*z + (1-f)z + g)^2
  //     float x = accDes.x;
  //     float y = accDes.y;
  //     float z = accDes.z - GRAVITY_MAGNITUDE;
  //     float g = GRAVITY_MAGNITUDE;
  //     float f = constrain(thrust_reduction_fairness, 0, 1);

  //     float r = 0;

  //     // solve as a quadratic
  //     float a = powf(x, 2) + powf(y, 2) + powf(z*f, 2);
  //     if (a<0) { a = 0; }

  //     float b = 2 * z*f*((1-f)*z + g);
  //     float c = powf(coll_max, 2) - powf((1-f)*z + g, 2);
  //     if (c<0) { c = 0; }

  //     if (fabsf(a)<1e-6f) {
  //       r = 0;
  //     } else {
  //       float sqrtterm = powf(b, 2) + 4.0f*a*c;
  //       r = (-b + sqrtf(sqrtterm))/(2.0f*a);
  //       r = constrain(r,0,1);
  //     }
  //     accDes.x = r*x;
  //     accDes.y = r*y;
  //     accDes.z = (r*f+(1-f))*z + g;
  //   }
  //   collCmd = constrain(accDes.z / R22, coll_min, coll_max);

  //   // FYI: this thrust will result in the accelerations
  //   // xdd = R02*coll
  //   // ydd = R12*coll

  //   // a unit vector pointing in the direction of the desired thrust (ie. the direction of body's z axis in the inertial frame)
  //   struct vec zI_des = vnormalize(accDes);

  //   // a unit vector pointing in the direction of the current thrust
  //   struct vec zI_cur = vnormalize(mkvec(R02, R12, R22));

  //   // a unit vector pointing in the direction of the inertial frame z-axis
  //   struct vec zI = mkvec(0, 0, 1);



  //   // ====== REDUCED ATTITUDE CONTROL ======

  //   // compute the error angle between the current and the desired thrust directions
  //   float dotProd = vdot(zI_cur, zI_des);
  //   dotProd = constrain(dotProd, -1, 1);
  //   float alpha = acosf(dotProd);

  //   // the axis around which this rotation needs to occur in the inertial frame (ie. an axis orthogonal to the two)
  //   struct vec rotAxisI = vzero();
  //   if (fabsf(alpha) > 1 * ARCMINUTE) {
  //     rotAxisI = vnormalize(vcross(zI_cur, zI_des));
  //   } else {
  //     rotAxisI = mkvec(1, 1, 0);
  //   }

  //   // the attitude error quaternion
  //   attErrorReduced.w = cosf(alpha / 2.0f);
  //   attErrorReduced.x = sinf(alpha / 2.0f) * rotAxisI.x;
  //   attErrorReduced.y = sinf(alpha / 2.0f) * rotAxisI.y;
  //   attErrorReduced.z = sinf(alpha / 2.0f) * rotAxisI.z;

  //   // choose the shorter rotation
  //   if (sinf(alpha / 2.0f) < 0) {
  //     rotAxisI = vneg(rotAxisI);
  //   }
  //   if (cosf(alpha / 2.0f) < 0) {
  //     rotAxisI = vneg(rotAxisI);
  //     attErrorReduced = qneg(attErrorReduced);
  //   }

  //   attErrorReduced = qnormalize(attErrorReduced);


  //   // ====== FULL ATTITUDE CONTROL ======

  //   // compute the error angle between the inertial and the desired thrust directions
  //   dotProd = vdot(zI, zI_des);
  //   dotProd = constrain(dotProd, -1, 1);
  //   alpha = acosf(dotProd);

  //   // the axis around which this rotation needs to occur in the inertial frame (ie. an axis orthogonal to the two)
  //   if (fabsf(alpha) > 1 * ARCMINUTE) {
  //     rotAxisI = vnormalize(vcross(zI, zI_des));
  //   } else {
  //     rotAxisI = mkvec(1, 1, 0);
  //   }

  //   // the quaternion corresponding to a roll and pitch around this axis
  //   struct quat attFullReqPitchRoll = mkquat(sinf(alpha / 2.0f) * rotAxisI.x,
  //                                            sinf(alpha / 2.0f) * rotAxisI.y,
  //                                            sinf(alpha / 2.0f) * rotAxisI.z,
  //                                            cosf(alpha / 2.0f));

  //   // the quaternion corresponding to a rotation to the desired yaw
  //   struct quat attFullReqYaw = mkquat(0, 0, sinf(radians(setpoint->attitude.yaw) / 2.0f), cosf(radians(setpoint->attitude.yaw) / 2.0f));

  //   // the full rotation (roll & pitch, then yaw)
  //   attDesiredFull = qqmul(attFullReqPitchRoll, attFullReqYaw);

  //   // back transform from the current attitude to get the error between rotations
  //   attErrorFull = qqmul(attitudeI, attDesiredFull);

  //   // correct rotation
  //   if (attErrorFull.w < 0) {
  //     attErrorFull = qneg(attErrorFull);
  //     attDesiredFull = qqmul(attitude, attErrorFull);
  //   }

  //   attErrorFull = qnormalize(attErrorFull);
  //   attDesiredFull = qnormalize(attDesiredFull);


  //   // ====== MIXING FULL & REDUCED CONTROL ======

  //   struct quat attError = qeye();

  //   if (mixing_factor <= 0) {
  //     // 100% reduced control (no yaw control)
  //     attError = attErrorReduced;
  //   } else if (mixing_factor >= 1) {
  //     // 100% full control (yaw controlled with same time constant as roll & pitch)
  //     attError = attErrorFull;
  //   } else {
  //     // mixture of reduced and full control

  //     // calculate rotation between the two errors
  //     temp1 = qinv(attErrorReduced);
  //     temp2 = qnormalize(qqmul(temp1, attErrorFull));

  //     // by defintion this rotation has the form [cos(alpha/2), 0, 0, sin(alpha/2)]
  //     // where the first element gives the rotation angle, and the last the direction
  //     alpha = 2.0f * acosf(constrain(temp2.w, -1, 1));

  //     // bisect the rotation from reduced to full control
  //     temp1 = mkquat(0,
  //                      0,
  //                      sinf(alpha * mixing_factor / 2.0f) * (temp2.z < 0 ? -1 : 1), // rotate in the correct direction
  //                      cosf(alpha * mixing_factor / 2.0f));

  //     attError = qnormalize(qqmul(attErrorReduced, temp1));
  //   }

    // ====== COMPUTE CONTROL SIGNALS ======

    // compute the commanded body rates
    // Chaoyi
    control_omega[0] = radians(setpoint->attitudeRate.roll);
    control_omega[1] = radians(setpoint->attitudeRate.pitch);
    control_omega[2] = radians(setpoint->attitudeRate.yaw);
    // control_omega[0] = 2.0f / tau_rp * attError.x;
    // control_omega[1] = 2.0f / tau_rp * attError.y;
    // control_omega[2] = 2.0f / tau_rp * attError.z + radians(setpoint->attitudeRate.yaw); // due to the mixing, this will behave with time constant tau_yaw

    // apply the rotation heuristic
    // if (control_omega[0] * omega[0] < 0 && fabsf(omega[0]) > heuristic_rp) { // desired rotational rate in direction opposite to current rotational rate
    //   control_omega[0] = omega_rp_max * (omega[0] < 0 ? -1 : 1); // maximum rotational rate in direction of current rotation
    // }

    // if (control_omega[1] * omega[1] < 0 && fabsf(omega[1]) > heuristic_rp) { // desired rotational rate in direction opposite to current rotational rate
    //   control_omega[1] = omega_rp_max * (omega[1] < 0 ? -1 : 1); // maximum rotational rate in direction of current rotation
    // }

    // if (control_omega[2] * omega[2] < 0 && fabsf(omega[2]) > heuristic_yaw) { // desired rotational rate in direction opposite to current rotational rate
    //   control_omega[2] = omega_rp_max * (omega[2] < 0 ? -1 : 1); // maximum rotational rate in direction of current rotation
    // }

    // // scale the commands to satisfy rate constraints
    // float scaling = 1;
    // scaling = fmax(scaling, fabsf(control_omega[0]) / omega_rp_max);
    // scaling = fmax(scaling, fabsf(control_omega[1]) / omega_rp_max);
    // scaling = fmax(scaling, fabsf(control_omega[2]) / omega_yaw_max);

    // control_omega[0] /= scaling;
    // control_omega[1] /= scaling;
    // control_omega[2] /= scaling;
    // Chaoyi
    control_thrust = setpoint->acceleration.z;
  }

  if (setpoint->mode.z == modeDisable) {
    control->thrustSi = 0.0f;
    control->torque[0] =  0.0f;
    control->torque[1] =  0.0f;
    control->torque[2] =  0.0f;
  } else {
    // control the body torques
    // struct vec omegaErr = mkvec((control_omega[0] - omega[0])/tau_rp_rate, 
    //                     (control_omega[1] - omega[1])/tau_rp_rate,
    //                     (control_omega[2] - omega[2])/tau_yaw_rate);

    struct vec omega_err = mkvec((control_omega[0] - omega[0]), 
                        (control_omega[1] - omega[1]),
                        (control_omega[2] - omega[2]));

    // omega gain  = [kR_xy, kR_xy, kR_z]
    // define vector omega_gain = [kR_xy, kR_xy, kR_z]
    struct vec omega_gain = mkvec(self->kR_xy, self->kR_xy, self->kR_z);

    struct vec alpha_desired = veltmul(omega_err, omega_gain);

    struct vec control_torque = mvmul(CRAZYFLIE_INERTIA, alpha_desired);

    // update the commanded body torques based on the current error in body rates
    // omegaErr = veltmul(omegaErr, omega_gain);
    

    // control_torque = mvmul(CRAZYFLIE_INERTIA, omegaErr);

    control->thrustSi = control_thrust * CF_MASS; // force to provide control_thrust
    control->torqueX = control_torque.x;
    control->torqueY = control_torque.y;
    control->torqueZ = control_torque.z;

    // printf("thrust: %.4f, torqueX: %.4f, torqueY: %.4f, torqueZ: %.4f\n", control->thrustSi, control->torqueX, control->torqueY, control->torqueZ);    
  }

  control->controlMode = controlModeForceTorque;
}


void controllerMellingerFirmwareInit(void)
{
  controllerMellingerInit(&g_self);
}

bool controllerMellingerFirmwareTest(void)
{
  return controllerMellingerTest(&g_self);
}

void controllerMellingerFirmware(control_t *control, const setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
{
  controllerMellinger(&g_self, control, setpoint, sensors, state, tick);
}


/**
 * Tunning variables for the full state Mellinger Controller
 */
PARAM_GROUP_START(ctrlMel)
/**
 * @brief Position P-gain (horizontal xy plane)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kp_xy, &g_self.kp_xy)
/**
 * @brief Position D-gain (horizontal xy plane)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kd_xy, &g_self.kd_xy)
/**
 * @brief Position I-gain (horizontal xy plane)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, ki_xy, &g_self.ki_xy)
/**
 * @brief Attitude maximum accumulated error (roll and pitch)
 */
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT, i_range_xy, &g_self.i_range_xy)
/**
 * @brief Position P-gain (vertical z plane)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kp_z, &g_self.kp_z)
/**
 * @brief Position D-gain (vertical z plane)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kd_z, &g_self.kd_z)
/**
 * @brief Position I-gain (vertical z plane)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, ki_z, &g_self.ki_z)
/**
 * @brief Position maximum accumulated error (vertical z plane)
 */
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT, i_range_z, &g_self.i_range_z)
/**
 * @brief total mass [kg]
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, mass, &g_self.mass)
/**
 * @brief Force to PWM stretch factor
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, massThrust, &g_self.massThrust)
/**
 * @brief Attitude P-gain (roll and pitch)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kR_xy, &g_self.kR_xy)
/**
 * @brief Attitude P-gain (yaw)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kR_z, &g_self.kR_z)
/**
 * @brief Attitude D-gain (roll and pitch)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kw_xy, &g_self.kw_xy)
/**
 * @brief Attitude D-gain (yaw)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kw_z, &g_self.kw_z)
/**
 * @brief Attitude I-gain (roll and pitch)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, ki_m_xy, &g_self.ki_m_xy)
/**
 * @brief Attitude I-gain (yaw)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, ki_m_z, &g_self.ki_m_z)
/**
 * @brief Angular velocity D-Gain (roll and pitch)
 */
PARAM_ADD_CORE(PARAM_FLOAT | PARAM_PERSISTENT, kd_omega_rp, &g_self.kd_omega_rp)
/**
 * @brief Attitude maximum accumulated error (roll and pitch)
 */
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT, i_range_m_xy, &g_self.i_range_m_xy)
/**
 * @brief Attitude maximum accumulated error (yaw)
 */
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT, i_range_m_z, &g_self.i_range_m_z)
PARAM_GROUP_STOP(ctrlMel)

/**
 * Logging variables for the command and reference signals for the
 * Mellinger controller
 */
LOG_GROUP_START(ctrlMel)
LOG_ADD(LOG_FLOAT, cmd_thrust, &g_self.cmd_thrust)
LOG_ADD(LOG_FLOAT, cmd_roll, &g_self.cmd_roll)
LOG_ADD(LOG_FLOAT, cmd_pitch, &g_self.cmd_pitch)
LOG_ADD(LOG_FLOAT, cmd_yaw, &g_self.cmd_yaw)
LOG_ADD(LOG_FLOAT, r_roll, &g_self.r_roll)
LOG_ADD(LOG_FLOAT, r_pitch, &g_self.r_pitch)
LOG_ADD(LOG_FLOAT, r_yaw, &g_self.r_yaw)
LOG_ADD(LOG_FLOAT, accelz, &g_self.accelz)
LOG_ADD(LOG_FLOAT, zdx, &g_self.z_axis_desired.x)
LOG_ADD(LOG_FLOAT, zdy, &g_self.z_axis_desired.y)
LOG_ADD(LOG_FLOAT, zdz, &g_self.z_axis_desired.z)
LOG_ADD(LOG_FLOAT, i_err_x, &g_self.i_error_x)
LOG_ADD(LOG_FLOAT, i_err_y, &g_self.i_error_y)
LOG_ADD(LOG_FLOAT, i_err_z, &g_self.i_error_z)
LOG_GROUP_STOP(ctrlMel)
