# MM5611 Advanced Robotics - RPRR Robot Analysis Project

A comprehensive robotics analysis project for a 4-DOF RPRR (Revolute-Prismatic-Revolute-Revolute) robot manipulator, developed as part of the MM5611 Advanced Robotics course.

## Table of Contents

- [Robot Description](#robot-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [GUI Applications](#gui-applications)
- [Simple Scripts](#simple-scripts)
- [Theoretical Background](#theoretical-background)
- [Documentation](#documentation)

---

## Robot Description

This project analyzes a 4 degree-of-freedom (4-DOF) robot manipulator with the following joint configuration:

| Joint | Type | Variable | Description |
|-------|------|----------|-------------|
| 1 | Revolute | θ₁ | Base rotation around Z-axis |
| 2 | Prismatic | d₂ | Vertical translation |
| 3 | Revolute | θ₃ | Shoulder rotation (α = π/2) |
| 4 | Revolute | θ₄ | Elbow rotation |

### Robot Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| L₀ | 0.5 m | Base height |
| L₁ | 0.4 m | Upper arm length |
| L₂ | 0.3 m | Forearm/tool length |
| d_max | 0.5 m | Maximum prismatic extension |

### Modified DH Parameters

| i | θᵢ | dᵢ | aᵢ | αᵢ |
|---|-----|-----|-----|------|
| 1 | θ₁ | L₀ | 0 | 0 |
| 2 | 0 | d₂ | 0 | 0 |
| 3 | θ₃ | 0 | 0 | π/2 |
| 4 | θ₄ | 0 | L₁ | 0 |

---

## Project Structure

```
MM5611-ILERI-ROBOTIK-PROJECT/
├── gui_forward.py      # Interactive Forward Kinematics GUI
├── gui_inverse.py      # Interactive Inverse Kinematics GUI
├── gui_dynamics.py     # Robot Dynamics Simulator (τ = M(q)q̈ + C(q,q̇)q̇ + g(q))
├── gui_dynamics2.py    # Extended Dynamics Simulator
├── jacobian_gui.py     # Force/Torque Analysis GUI (τ = Jᵀ·F)
├── trajectory_gui.py   # Trajectory Animation (Line, Circle, Sinusoidal)
├── forward_simple.py   # Simple forward kinematics script
├── inverse_simple.py   # Simple inverse kinematics script
├── requirements.txt    # Python dependencies
├── pdf/
│   └── report.pdf      # Detailed theoretical report
└── README.md           # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to the project directory
cd MM5611-ILERI-ROBOTIK-PROJECT

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- **roboticstoolbox-python** (v1.1.1) - Peter Corke's Robotics Toolbox
- **numpy** (<2.0) - Numerical computing
- **matplotlib** - Visualization and GUI
- **scipy** - Scientific computing
- **spatialmath-python** - Spatial mathematics

---

## GUI Applications

### 1. Forward Kinematics GUI (`gui_forward.py`)

Interactive 3D visualization with real-time joint control.

```bash
python gui_forward.py
```

**Features:**
- Sliders and text boxes for all 4 joint variables
- Editable robot parameters (L₀, L₁, L₂, d_max)
- Real-time 3D robot visualization
- End-effector position display (X, Y, Z)
- Workspace visualization toggle
- Coordinate frame display at each joint

---

### 2. Inverse Kinematics GUI (`gui_inverse.py`)

Compute joint values from target end-effector position.

```bash
python gui_inverse.py
```

**Features:**
- Enter target X, Y, Z coordinates
- Analytical and numerical IK solvers
- Multiple solution navigation (elbow up/down)
- Solution verification with error display
- Target position visualization with crosshairs

---

### 3. Jacobian & Force Analysis GUI (`jacobian_gui.py`)

Analyze the relationship between joint torques and end-effector forces.

```bash
python jacobian_gui.py
```

**Features:**
- Interactive joint configuration control
- Apply end-effector forces (Fx, Fy, Fz)
- Real-time Jacobian matrix display
- Joint torque/force computation (τ = Jᵀ·F)
- Force vector visualization on robot

---

### 4. Dynamics Simulator (`gui_dynamics.py`, `gui_dynamics2.py`)

Visualize the manipulator dynamics equation: **τ = M(q)q̈ + C(q,q̇)q̇ + g(q)**

```bash
python gui_dynamics.py
# or
python gui_dynamics2.py
```

**Features:**
- Joint position, velocity, and acceleration control
- Mass matrix M(q) visualization
- Coriolis/centrifugal matrix C(q,q̇) display
- Gravity vector g(q) computation
- Required torque calculation

**Mass Properties:**
| Link | Mass | Inertia |
|------|------|---------|
| Prismatic slider | 2.0 kg | - |
| Upper arm | 1.5 kg | (1/12)·m·L₁² |
| Forearm + tool | 0.8 kg | (1/12)·m·L₂² |

---

### 5. Trajectory Animation (`trajectory_gui.py`)

Animate the robot following predefined trajectories.

```bash
python trajectory_gui.py
```

**Trajectory Types:**
- **Straight Line** - Diagonal path in 3D space
- **Circular Arc** - Arc in XY plane with height variation
- **Sinusoidal** - Wave pattern with rotation

**Features:**
- Play/Pause/Reset controls
- Trajectory type selection
- Cubic polynomial time scaling
- Path visualization
- Real-time joint value display

---

## Simple Scripts

### Forward Kinematics (`forward_simple.py`)

Quick computation of end-effector position from joint values.

```bash
python forward_simple.py
```

**Usage:** Edit the joint values in the script:
```python
theta1 = 0      # degrees
d2 = 0.2        # meters
theta3 = 0      # degrees
theta4 = 45     # degrees
```

**Output:**
```
========================================
  FORWARD KINEMATICS
========================================
  Joint Values:
    θ1 = 0.00°
    d2 = 0.2000 m
    θ3 = 0.00°
    θ4 = 45.00°

  End-Effector Position:
    X = 0.4950 m
    Y = 0.0000 m
    Z = 0.9828 m
========================================
```

---

### Inverse Kinematics (`inverse_simple.py`)

Compute joint values for a target position using numerical IK.

```bash
python inverse_simple.py
```

**Usage:** Edit the target position in the script:
```python
x_target = 0.7
y_target = 0.0
z_target = 0.7
```

---

## Theoretical Background

### Forward Kinematics

The end-effector position is derived from the compound transformation matrix T₀⁵:

```
x = cos(θ₁) · (L₁·cos(θ₃) + L₂·cos(θ₃ + θ₄))
y = sin(θ₁) · (L₁·cos(θ₃) + L₂·cos(θ₃ + θ₄))
z = L₀ + d₂ + L₁·sin(θ₃) + L₂·sin(θ₃ + θ₄)
```

**Key Observations:**
- θ₁ controls only the azimuthal angle in the XY plane
- θ₃ and θ₄ form a 2-DOF planar arm in the vertical plane
- d₂ provides pure vertical translation
- The robot has cylindrical symmetry about the Z-axis

---

### Inverse Kinematics

**Problem:** 4 unknowns but only 3 position equations → **redundant system**

**Solution Algorithm:**
1. **θ₁ = atan2(y, x)** - Base rotation to face target
2. **r = √(x² + y²)** - Horizontal reach
3. **Choose d₂** - Redundancy resolution
4. **h = z - L₀ - d₂** - Height above joint 3
5. **Solve 2R planar arm** for θ₃ and θ₄ using law of cosines

**Multiple Solutions:** Two configurations exist (elbow up/down) for each valid d₂.

---

### Jacobian Analysis

The translational Jacobian relates joint velocities to end-effector velocity: **v = J·q̇**

**Key Findings:**
- **Redundancy:** Columns 1 and 3 of J are identical (J_col1 = J_col3) because θ₁ and θ₃ rotate about parallel axes
- **Singularity:** When θ₄ = ±90°, the Jacobian rank drops to 2

**Static Force Analysis:** τ = Jᵀ·F

---

### Workspace Analysis

The workspace is analyzed in cylindrical coordinates. The cross-section is defined by:

$$r² + (z - (L₀ + d₂))² = (L₁ + L₂)²$$

**Void Condition:** If (d₂_max - d₂_min) < 2(L₁ + L₂), a hollow region exists inside the workspace.

---


