/**
 *    ||          ____  _ __
 * +------+      / __ )(_) /_______________ _____  ___
 * | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2011-2022 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>
 *
 * power_distribution_quadrotor.c - Crazyflie stock power distribution code
 */

#include "power_distribution.h"

#include <string.h>
#include "log.h"
#include "param.h"
#include "num.h"
#include "autoconf.h"
#include "config.h"
#include "math.h"

#ifndef CONFIG_MOTORS_DEFAULT_IDLE_THRUST
#define DEFAULT_IDLE_THRUST 0
#else
#define DEFAULT_IDLE_THRUST CONFIG_MOTORS_DEFAULT_IDLE_THRUST
#endif

static uint32_t idleThrust = DEFAULT_IDLE_THRUST;
static float armLength = 0.046f; // m;
static float thrustToTorque = 0.005964552f;

// thrust = a * pwm^2 + b * pwm
// static float pwmToThrustA = 0.091492681f;
// static float pwmToThrustB = 0.067673604f;
static float pwmToThrustA = 0.11459406f;
static float pwmToThrustB = 0.02580751f;

// variable to log here
static float cmd_pwm1 = 0.0f;
static float cmd_pwm2 = 0.0f;
static float cmd_pwm3 = 0.0f;
static float cmd_pwm4 = 0.0f;
static float cmd_f1 = 0.0f;
static float cmd_f2 = 0.0f;
static float cmd_f3 = 0.0f;
static float cmd_f4 = 0.0f;

int powerDistributionMotorType(uint32_t id)
{
  return 1;
}

uint16_t powerDistributionStopRatio(uint32_t id)
{
  return 0;
}

void powerDistributionInit(void)
{
}

bool powerDistributionTest(void)
{
  bool pass = true;
  return pass;
}

static uint16_t capMinThrust(float thrust, uint32_t minThrust)
{
  if (thrust < minThrust)
  {
    return minThrust;
  }

  return thrust;
}

static void powerDistributionLegacy(const control_t *control, motors_thrust_uncapped_t *motorThrustUncapped)
{
  int16_t r = control->roll / 2.0f;
  int16_t p = control->pitch / 2.0f;

  motorThrustUncapped->motors.m1 = control->thrust - r + p + control->yaw;
  motorThrustUncapped->motors.m2 = control->thrust - r - p - control->yaw;
  motorThrustUncapped->motors.m3 = control->thrust + r - p + control->yaw;
  motorThrustUncapped->motors.m4 = control->thrust + r + p - control->yaw;

  cmd_pwm1 = (float)motorThrustUncapped->motors.m1 / (float)UINT16_MAX;
  cmd_pwm2 = (float)motorThrustUncapped->motors.m2 / (float)UINT16_MAX;
  cmd_pwm3 = (float)motorThrustUncapped->motors.m3 / (float)UINT16_MAX;
  cmd_pwm4 = (float)motorThrustUncapped->motors.m4 / (float)UINT16_MAX;
}

static void powerDistributionForceTorque(const control_t *control, motors_thrust_uncapped_t *motorThrustUncapped)
{
  static float motorForces[STABILIZER_NR_OF_MOTORS];

  const float arm = 0.707106781f * armLength;
  const float rollPart = 0.25f / arm * control->torqueX;
  const float pitchPart = 0.25f / arm * control->torqueY;
  const float thrustPart = 0.25f * control->thrustSi; // N (per rotor)
  const float yawPart = 0.25f * control->torqueZ / thrustToTorque;

  motorForces[0] = thrustPart - rollPart - pitchPart - yawPart;
  motorForces[1] = thrustPart - rollPart + pitchPart + yawPart;
  motorForces[2] = thrustPart + rollPart + pitchPart - yawPart;
  motorForces[3] = thrustPart + rollPart - pitchPart + yawPart;

  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++)
  {
    float motorForce = motorForces[motorIndex];
    if (motorForce < 0.0f)
    {
      motorForce = 0.0f;
    }

    float motor_pwm = (-pwmToThrustB + sqrtf(pwmToThrustB * pwmToThrustB + 4.0f * pwmToThrustA * motorForce)) / (2.0f * pwmToThrustA);

    // add to log here
    switch (motorIndex)
    {
    case 0:
      cmd_pwm1 = motor_pwm;
      cmd_f1 = motorForce;
      break;
    case 1:
      cmd_pwm2 = motor_pwm;
      cmd_f2 = motorForce;
      break;
    case 2:
      cmd_pwm3 = motor_pwm;
      cmd_f3 = motorForce;
      break;
    case 3:
      cmd_pwm4 = motor_pwm;
      cmd_f4 = motorForce;
      break;
    }

    motorThrustUncapped->list[motorIndex] = motor_pwm * UINT16_MAX;
  }
}

static void powerDistributionForce(const control_t *control, motors_thrust_uncapped_t* motorThrustUncapped) {
  // NOTE: now this only works for upgrade bundle
  static float motorForces[STABILIZER_NR_OF_MOTORS];

  const float arm = 0.707106781f * armLength;
  const float rollPart = 0.25f / arm * control->tau_x;
  const float pitchPart = 0.25f / arm * control->tau_y;
  const float thrustPart = 0.25f * control->T; // N (per rotor)
  const float yawPart = 0.25f * control->tau_z / thrustToTorque;

//   motorForces[0] = thrustPart - rollPart - pitchPart - yawPart;
//   motorForces[1] = thrustPart - rollPart + pitchPart + yawPart;
//   motorForces[2] = thrustPart + rollPart + pitchPart - yawPart;
//   motorForces[3] = thrustPart + rollPart - pitchPart + yawPart;

  motorForces[0] = thrustPart - rollPart + pitchPart + yawPart;
  motorForces[1] = thrustPart - rollPart - pitchPart - yawPart;
  motorForces[2] = thrustPart + rollPart - pitchPart + yawPart;
  motorForces[3] = thrustPart + rollPart + pitchPart - yawPart;

  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++) {
    float motorForce = motorForces[motorIndex];
    if (motorForce < 0.0f) {
      motorForce = 0.0f;
    }

    // NOTE: here, I use simple linear model to convert force to pwm. need further identification
    // mass to pwm ratio = 85000, force=1.0/4 N, pwm=85000
    float motor_pwm = motorForce * 5.188f; // / 0.25f * 85000.0f / 65535.0f; // range: 0 - 1.0

    // float motor_pwm = (-pwmToThrustB + sqrtf(pwmToThrustB * pwmToThrustB + 4.0f * pwmToThrustA * motorForce)) / (2.0f * pwmToThrustA);

    // add to log here
    switch (motorIndex)
    {
    case 0:
      cmd_pwm1 = motor_pwm;
      cmd_f1 = motorForces[motorIndex];
      break;
    case 1:
      cmd_pwm2 = motor_pwm;
      cmd_f2 = motorForces[motorIndex];
      break;
    case 2:
      cmd_pwm3 = motor_pwm;
      cmd_f3 = motorForces[motorIndex];
      break;
    case 3:
      cmd_pwm4 = motor_pwm;
      cmd_f4 = motorForces[motorIndex];
      break;
    }

    motorThrustUncapped->list[motorIndex] = motor_pwm * UINT16_MAX;
  }
}

void powerDistribution(const control_t *control, motors_thrust_uncapped_t *motorThrustUncapped)
{
  switch (control->controlMode)
  {
  case controlModeLegacy:
    powerDistributionLegacy(control, motorThrustUncapped);
    break;
  case controlModeForceTorque:
    powerDistributionForceTorque(control, motorThrustUncapped);
    break;
  case controlModeForce:
    powerDistributionForce(control, motorThrustUncapped);
    break;
  default:
    // Nothing here
    break;
  }
}

void powerDistributionCap(const motors_thrust_uncapped_t *motorThrustBatCompUncapped, motors_thrust_pwm_t *motorPwm)
{
  const int32_t maxAllowedThrust = UINT16_MAX;

  // Find highest thrust
  int32_t highestThrustFound = 0;
  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++)
  {
    const int32_t thrust = motorThrustBatCompUncapped->list[motorIndex];
    if (thrust > highestThrustFound)
    {
      highestThrustFound = thrust;
    }
  }

  int32_t reduction = 0;
  if (highestThrustFound > maxAllowedThrust)
  {
    reduction = highestThrustFound - maxAllowedThrust;
  }

  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++)
  {
    int32_t thrustCappedUpper = motorThrustBatCompUncapped->list[motorIndex] - reduction;
    motorPwm->list[motorIndex] = capMinThrust(thrustCappedUpper, idleThrust);
  }
}

/**
 * power distribution parameters
 */
LOG_GROUP_START(power)
LOG_ADD(LOG_FLOAT, cmd_pwm1, &cmd_pwm1)
LOG_ADD(LOG_FLOAT, cmd_pwm2, &cmd_pwm2)
LOG_ADD(LOG_FLOAT, cmd_pwm3, &cmd_pwm3)
LOG_ADD(LOG_FLOAT, cmd_pwm4, &cmd_pwm4)
LOG_ADD(LOG_FLOAT, cmd_f1, &cmd_f1)
LOG_ADD(LOG_FLOAT, cmd_f2, &cmd_f2)
LOG_ADD(LOG_FLOAT, cmd_f3, &cmd_f3)
LOG_ADD(LOG_FLOAT, cmd_f4, &cmd_f4)
LOG_GROUP_STOP(power)

/**
 * Power distribution parameters
 */
PARAM_GROUP_START(powerDist)
/**
 * @brief Motor thrust to set at idle (default: 0)
 *
 * This is often needed for brushless motors as
 * it takes time to start up the motor. Then a
 * common value is between 3000 - 6000.
 */
PARAM_ADD_CORE(PARAM_UINT32 | PARAM_PERSISTENT, idleThrust, &idleThrust)
PARAM_GROUP_STOP(powerDist)

/**
 * System identification parameters for quad rotor
 */
PARAM_GROUP_START(quadSysId)

PARAM_ADD(PARAM_FLOAT, thrustToTorque, &thrustToTorque)
PARAM_ADD(PARAM_FLOAT, pwmToThrustA, &pwmToThrustA)
PARAM_ADD(PARAM_FLOAT, pwmToThrustB, &pwmToThrustB)

/**
 * @brief Length of arms (m)
 *
 * The distance from the center to a motor
 */
PARAM_ADD(PARAM_FLOAT, armLength, &armLength)
PARAM_GROUP_STOP(quadSysId)
