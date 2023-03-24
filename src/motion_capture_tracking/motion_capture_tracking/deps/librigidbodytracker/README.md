[![CMake](https://github.com/IMRCLab/librigidbodytracker/actions/workflows/cmake.yml/badge.svg)](https://github.com/IMRCLab/librigidbodytracker/actions/workflows/cmake.yml)

# librigidbodytracker
This library helps to track (i.e. estimate the pose) of rigid-bodies.
It assumes that an initial estimate for the pose of each rigid body is given.
The new poses are estimated using the iterative closest point algorithm (ICP) frame-by-frame.

The library is used in the Crazyswarm project.
