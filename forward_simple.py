"""
RPRR Robot - Simple Forward Kinematics
Input joint values, get end-effector position
"""

import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
import numpy as np
from math import pi

# Robot parameters
L0 = 0.5  # Base height
L1 = 0.4  # Link 1
L2 = 0.3  # End-effector
dmax = 0.5  # Max prismatic extension

# Build robot
robot = rtb.DHRobot([
    RevoluteMDH(a=0, alpha=0, d=L0),
    PrismaticMDH(a=0, alpha=0, theta=0, qlim=[0, dmax]),
    RevoluteMDH(a=0, alpha=pi/2, d=0),
    RevoluteMDH(a=L1, alpha=0, d=0),
], tool=SE3.Tx(L2))

# ============================================
# INPUT: Joint values
# ============================================
theta1 = 0      # degrees
d2 = 0.2        # meters
theta3 = 0      # degrees
theta4 = 45     # degrees

# ============================================
# Convert and compute
# ============================================
q = [np.radians(theta1), d2, np.radians(theta3), np.radians(theta4)]
T = robot.fkine(q)

# ============================================
# OUTPUT
# ============================================
print("\n" + "="*40)
print("  FORWARD KINEMATICS")
print("="*40)
print(f"\n  Joint Values:")
print(f"    θ1 = {theta1:.2f}°")
print(f"    d2 = {d2:.4f} m")
print(f"    θ3 = {theta3:.2f}°")
print(f"    θ4 = {theta4:.2f}°")
print(f"\n  End-Effector Position:")
print(f"    X = {T.t[0]:.4f} m")
print(f"    Y = {T.t[1]:.4f} m")
print(f"    Z = {T.t[2]:.4f} m")
print("="*40 + "\n")
