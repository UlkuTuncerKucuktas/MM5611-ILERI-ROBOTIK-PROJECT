"""
RPRR Robot - Simple Inverse Kinematics
Input target position, get joint values
"""

import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
import numpy as np
from math import pi, atan2, sqrt

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
# INPUT: Target position (meters)
# ============================================
x_target = 0.7
y_target = 0.0
z_target = 0.7

# ============================================
# Solve IK (numerical)
# ============================================
target = SE3.Trans(x_target, y_target, z_target)

# Try multiple initial guesses
best_solution = None
best_error = float('inf')

for _ in range(20):
    q0 = np.random.uniform([-pi, 0, -pi, -pi], [pi, dmax, pi, pi])
    sol = robot.ikine_LM(target, q0=q0, mask=[1,1,1,0,0,0])
    if sol.success:
        T = robot.fkine(sol.q)
        error = np.linalg.norm(T.t - np.array([x_target, y_target, z_target]))
        if error < best_error:
            best_error = error
            best_solution = sol.q

# ============================================
# OUTPUT
# ============================================
print("\n" + "="*40)
print("  INVERSE KINEMATICS")
print("="*40)
print(f"\n  Target Position:")
print(f"    X = {x_target:.4f} m")
print(f"    Y = {y_target:.4f} m")
print(f"    Z = {z_target:.4f} m")

if best_solution is not None and best_error < 0.01:
    print(f"\n  Joint Values (Solution Found):")
    print(f"    θ1 = {np.degrees(best_solution[0]):.2f}°")
    print(f"    d2 = {best_solution[1]:.4f} m")
    print(f"    θ3 = {np.degrees(best_solution[2]):.2f}°")
    print(f"    θ4 = {np.degrees(best_solution[3]):.2f}°")
    print(f"\n  Error: {best_error*1000:.3f} mm")
else:
    print(f"\n  No solution found!")
    print(f"  Target may be outside workspace.")

print("="*40 + "\n")
