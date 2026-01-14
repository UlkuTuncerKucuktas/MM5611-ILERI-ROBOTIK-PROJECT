"""
RPRR Robot - Inverse Kinematics Simulator
Enter target position, compute joint values
"""

import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
import numpy as np
from math import pi, atan2, sqrt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

# ============================================
# Color Palette
# ============================================
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#1f3460',
    'accent1': '#e94560',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0a0',
    'link1': '#ff6b6b',
    'link2': '#4ecdc4',
    'link3': '#ffe66d',
    'link4': '#c44dff',
    'link5': '#ff9f43',
    'grid': '#2d4a7c',
    'joint': '#ffffff',
    'end_effector': '#e94560',
    'target': '#00ff88',
    'param': '#00d9a0',
    'ik_input': '#ff9f43',
}

# ============================================
# Robot Parameters
# ============================================
params = {
    'L0': 0.5,
    'L1': 0.4,
    'L2': 0.3,
    'dmax': 0.5,
}

# ============================================
# Robot Builder
# ============================================
def build_robot(L0, L1, L2, dmax):
    return rtb.DHRobot(
        [
            RevoluteMDH(a=0, alpha=0, d=L0, qlim=[-pi, pi]),
            PrismaticMDH(a=0, alpha=0, theta=0, qlim=[0, dmax]),
            RevoluteMDH(a=0, alpha=pi/2, d=0, qlim=[-pi, pi]),
            RevoluteMDH(a=L1, alpha=0, d=0, qlim=[-pi, pi]),
        ],
        name="RPRR Robot",
        tool=SE3.Tx(L2)
    )

robot = build_robot(params['L0'], params['L1'], params['L2'], params['dmax'])

# ============================================
# Analytical IK for RPRR
# ============================================
def ik_rprr(x, y, z, L0, L1, L2, dmax):
    """
    Analytical inverse kinematics for RPRR robot.
    Returns list of solutions (may have multiple or none).
    Each solution is [theta1, d2, theta3, theta4] in radians/meters.
    """
    solutions = []
    
    # theta1: rotation around Z to point towards target
    theta1 = atan2(y, x)
    
    # Distance in XY plane from origin
    r_xy = sqrt(x**2 + y**2)
    
    # The arm (L1 + L2) must reach from the prismatic joint output
    # to the target point. The prismatic joint is at height L0 + d2.
    
    # For each theta3, theta4 combination, we get different reach
    # Let's solve for the case where theta3 controls elevation
    # and theta4 controls the final orientation
    
    # Simplified approach: 
    # After theta1 rotation, we work in the r-z plane where r = sqrt(x^2 + y^2)
    # The end-effector position relative to joint 3 depends on theta3 and theta4
    
    # Try multiple configurations
    for theta3_sign in [1, -1]:
        for theta4_sign in [1, -1]:
            try:
                # The geometry after joint 2:
                # Joint 3 rotates about an axis that's been tilted 90° (alpha=pi/2)
                # This means theta3 rotates in the r-z plane
                
                # Distance from joint 2 to end-effector in r-z plane
                # needs to equal sqrt((r_xy)^2 + (z - L0 - d2)^2)
                
                # For a given d2, the arm (L1, L2) forms a 2R mechanism
                # r_target = r_xy
                # z_target = z - L0 - d2
                
                # Try different d2 values
                for d2 in np.linspace(0, dmax, 10):
                    z_rel = z - L0 - d2
                    r_target = r_xy
                    
                    # Distance to reach
                    D = sqrt(r_target**2 + z_rel**2)
                    
                    # Check if reachable by L1 + L2
                    if D > (L1 + L2) or D < abs(L1 - L2):
                        continue
                    
                    # 2R IK in the r-z plane
                    # cos(theta4) from law of cosines
                    cos_t4 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
                    if abs(cos_t4) > 1:
                        continue
                    
                    theta4 = theta4_sign * np.arccos(cos_t4)
                    
                    # theta3 from geometry
                    beta = atan2(z_rel, r_target)
                    alpha_angle = atan2(L2 * np.sin(theta4), L1 + L2 * np.cos(theta4))
                    theta3 = beta - alpha_angle
                    
                    # Validate solution with forward kinematics
                    q = [theta1, d2, theta3, theta4]
                    T = robot.fkine(q)
                    pos = T.t
                    
                    error = sqrt((pos[0]-x)**2 + (pos[1]-y)**2 + (pos[2]-z)**2)
                    
                    if error < 0.01:  # Within 1cm
                        # Check joint limits
                        if 0 <= d2 <= dmax and -pi <= theta3 <= pi and -pi <= theta4 <= pi:
                            solutions.append([theta1, d2, theta3, theta4])
                            
            except:
                continue
    
    # Remove duplicates
    unique_solutions = []
    for sol in solutions:
        is_dup = False
        for usol in unique_solutions:
            if all(abs(s1 - s2) < 0.01 for s1, s2 in zip(sol, usol)):
                is_dup = True
                break
        if not is_dup:
            unique_solutions.append(sol)
    
    return unique_solutions

def ik_numerical(x, y, z):
    """Numerical IK using robotics toolbox"""
    target = SE3.Trans(x, y, z)
    
    # Try multiple initial guesses
    best_solution = None
    best_error = float('inf')
    
    for _ in range(10):
        q0 = np.random.uniform([-pi, 0, -pi, -pi], [pi, params['dmax'], pi, pi])
        try:
            sol = robot.ikine_LM(target, q0=q0, mask=[1,1,1,0,0,0])
            if sol.success:
                T = robot.fkine(sol.q)
                error = np.linalg.norm(T.t - np.array([x, y, z]))
                if error < best_error:
                    best_error = error
                    best_solution = sol.q
        except:
            pass
    
    return best_solution, best_error

# ============================================
# Figure Setup
# ============================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(COLORS['bg_dark'])

ax = fig.add_subplot(121, projection='3d')
ax.set_position([0.02, 0.08, 0.55, 0.88])
ax.set_facecolor(COLORS['bg_medium'])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(COLORS['grid'])
ax.yaxis.pane.set_edgecolor(COLORS['grid'])
ax.zaxis.pane.set_edgecolor(COLORS['grid'])

# Current state
current_q = [0.0, 0.2, 0.0, 0.0]
target_pos = [0.7, 0.0, 0.7]  # Initial target
ik_solutions = []
current_solution_idx = 0

# ============================================
# Control Panel
# ============================================
panel_ax = fig.add_axes([0.58, 0.02, 0.40, 0.96])
panel_ax.set_facecolor(COLORS['bg_medium'])
panel_ax.set_xticks([])
panel_ax.set_yticks([])
for spine in panel_ax.spines.values():
    spine.set_color(COLORS['bg_light'])
    spine.set_linewidth(2)

fig.text(0.78, 0.96, 'INVERSE KINEMATICS', fontsize=16, fontweight='bold',
         ha='center', color=COLORS['text_primary'])

# ============================================
# TARGET POSITION Section
# ============================================
fig.text(0.78, 0.91, '──── TARGET POSITION ────', fontsize=9,
         ha='center', color=COLORS['text_secondary'])

target_rows = [0.85, 0.79, 0.73]
target_labels = ['X', 'Y', 'Z']
target_colors = ['#ff6b6b', '#4ecdc4', '#74b9ff']
target_textboxes = []

for i, (row, label, color) in enumerate(zip(target_rows, target_labels, target_colors)):
    fig.text(0.62, row + 0.015, f'{label} target:', fontsize=11, 
             color=color, fontweight='bold', ha='left')
    
    ax_text = plt.axes([0.72, row, 0.12, 0.04])
    ax_text.set_facecolor(COLORS['bg_dark'])
    for spine in ax_text.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
    
    textbox = TextBox(ax_text, '', initial=f'{target_pos[i]:.3f}',
                     color=COLORS['bg_dark'], hovercolor=COLORS['bg_light'])
    target_textboxes.append(textbox)
    
    fig.text(0.85, row + 0.015, 'm', fontsize=10, color=COLORS['text_secondary'])

t_x, t_y, t_z = target_textboxes

# Solve IK Button
ax_solve = plt.axes([0.65, 0.65, 0.26, 0.05])
btn_solve = Button(ax_solve, 'SOLVE IK', color=COLORS['target'], hovercolor='#00cc6a')
btn_solve.label.set_color('black')
btn_solve.label.set_fontweight('bold')
btn_solve.label.set_fontsize(12)

# ============================================
# IK RESULT Section
# ============================================
fig.text(0.78, 0.58, '──── IK SOLUTION ────', fontsize=9,
         ha='center', color=COLORS['text_secondary'])

# Solution info box
ax_sol_info = plt.axes([0.62, 0.44, 0.32, 0.12])
ax_sol_info.set_facecolor(COLORS['bg_dark'])
ax_sol_info.set_xticks([])
ax_sol_info.set_yticks([])
for spine in ax_sol_info.spines.values():
    spine.set_color(COLORS['param'])
    spine.set_linewidth(1.5)

text_solution = ax_sol_info.text(0.5, 0.5, 'No solution computed', 
                                  ha='center', va='center', fontsize=10,
                                  color=COLORS['text_primary'], 
                                  transform=ax_sol_info.transAxes,
                                  family='monospace')

# Next/Prev solution buttons
ax_prev = plt.axes([0.62, 0.38, 0.15, 0.04])
btn_prev = Button(ax_prev, '< PREV', color=COLORS['bg_light'], hovercolor=COLORS['accent1'])
btn_prev.label.set_color('white')
btn_prev.label.set_fontsize(9)

ax_next = plt.axes([0.79, 0.38, 0.15, 0.04])
btn_next = Button(ax_next, 'NEXT >', color=COLORS['bg_light'], hovercolor=COLORS['accent1'])
btn_next.label.set_color('white')
btn_next.label.set_fontsize(9)

# Solution counter
text_sol_counter = fig.text(0.78, 0.355, '', fontsize=9, ha='center', 
                            color=COLORS['text_secondary'])

# ============================================
# ROBOT PARAMETERS Section
# ============================================
fig.text(0.78, 0.30, '──── ROBOT PARAMETERS ────', fontsize=9,
         ha='center', color=COLORS['text_secondary'])

param_rows = [0.24, 0.18, 0.12, 0.06]
param_labels = ['L0', 'L1', 'L2', 'dmax']
param_color = COLORS['param']

param_textboxes = []

for i, (row, label) in enumerate(zip(param_rows, param_labels)):
    fig.text(0.62, row + 0.012, f'{label}:', fontsize=10, 
             color=param_color, fontweight='bold', ha='left')
    
    ax_text = plt.axes([0.70, row, 0.08, 0.03])
    ax_text.set_facecolor(COLORS['bg_dark'])
    for spine in ax_text.spines.values():
        spine.set_color(param_color)
        spine.set_linewidth(1.5)
    
    key = label.lower() if label != 'dmax' else 'dmax'
    if label == 'L0': key = 'L0'
    elif label == 'L1': key = 'L1'
    elif label == 'L2': key = 'L2'
    
    textbox = TextBox(ax_text, '', initial=f'{params[param_labels[i]]:.2f}',
                     color=COLORS['bg_dark'], hovercolor=COLORS['bg_light'])
    param_textboxes.append(textbox)
    
    fig.text(0.79, row + 0.012, 'm', fontsize=9, color=COLORS['text_secondary'])

t_L0, t_L1, t_L2, t_dmax = param_textboxes

# Error display
ax_error = plt.axes([0.82, 0.06, 0.12, 0.21])
ax_error.set_facecolor(COLORS['bg_dark'])
ax_error.set_xticks([])
ax_error.set_yticks([])
for spine in ax_error.spines.values():
    spine.set_color(COLORS['accent1'])
    spine.set_linewidth(1.5)

fig.text(0.88, 0.265, 'Error', fontsize=9, ha='center', color=COLORS['text_secondary'])
text_error = ax_error.text(0.5, 0.5, '---', ha='center', va='center', fontsize=12,
                           color=COLORS['accent1'], transform=ax_error.transAxes,
                           fontweight='bold', family='monospace')

# ============================================
# Drawing Functions
# ============================================
def get_link_positions(q):
    positions = [[0, 0, 0]]
    T_all = robot.fkine_all(q)
    for T in T_all:
        positions.append(T.t.tolist())
    T_tool = robot.fkine(q)
    positions.append(T_tool.t.tolist())
    return np.array(positions)

def draw_robot(q_rad, target=None):
    ax.cla()
    ax.set_facecolor(COLORS['bg_medium'])
    
    pos = get_link_positions(q_rad)
    max_reach = params['L0'] + params['dmax'] + params['L1'] + params['L2'] + 0.3
    
    # Floor grid
    floor_size = max_reach * 0.8
    for i in np.linspace(-floor_size, floor_size, 9):
        alpha = 0.3 - 0.15 * abs(i) / floor_size
        ax.plot3D([i, i], [-floor_size, floor_size], [0, 0], 
                  color=COLORS['grid'], alpha=alpha, linewidth=0.8)
        ax.plot3D([-floor_size, floor_size], [i, i], [0, 0], 
                  color=COLORS['grid'], alpha=alpha, linewidth=0.8)
    
    # Draw target position
    if target is not None:
        ax.scatter3D([target[0]], [target[1]], [target[2]], 
                     c=COLORS['target'], s=500, marker='*', 
                     edgecolors='white', linewidths=2, zorder=10, alpha=0.9)
        # Target crosshairs
        size = 0.1
        ax.plot3D([target[0]-size, target[0]+size], [target[1], target[1]], [target[2], target[2]],
                  color=COLORS['target'], linewidth=2, alpha=0.7)
        ax.plot3D([target[0], target[0]], [target[1]-size, target[1]+size], [target[2], target[2]],
                  color=COLORS['target'], linewidth=2, alpha=0.7)
        ax.plot3D([target[0], target[0]], [target[1], target[1]], [target[2]-size, target[2]+size],
                  color=COLORS['target'], linewidth=2, alpha=0.7)
    
    # Base
    theta_base = np.linspace(0, 2*pi, 40)
    r_base = 0.12
    ax.plot3D(r_base * np.cos(theta_base), r_base * np.sin(theta_base), 
              np.zeros(40), color=COLORS['accent1'], linewidth=4, alpha=0.8)
    
    # Links
    link_colors = [COLORS['link1'], COLORS['link2'], COLORS['link3'], 
                   COLORS['link4'], COLORS['link5']]
    
    for i in range(len(pos) - 1):
        color = link_colors[i % len(link_colors)]
        ax.plot3D([pos[i, 0], pos[i+1, 0]], 
                  [pos[i, 1], pos[i+1, 1]], 
                  [pos[i, 2], pos[i+1, 2]], 
                  color=color, linewidth=14, alpha=0.3, solid_capstyle='round')
        ax.plot3D([pos[i, 0], pos[i+1, 0]], 
                  [pos[i, 1], pos[i+1, 1]], 
                  [pos[i, 2], pos[i+1, 2]], 
                  color=color, linewidth=8, solid_capstyle='round')
    
    # Joints
    ax.scatter3D(pos[:-1, 0], pos[:-1, 1], pos[:-1, 2], 
                 c=COLORS['joint'], s=250, marker='o', 
                 edgecolors=COLORS['accent1'], linewidths=3, zorder=5, alpha=0.95)
    
    # End-effector
    ax.scatter3D([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], 
                 c=COLORS['end_effector'], s=400, marker='D', 
                 edgecolors='white', linewidths=3, zorder=6)
    
    # Coordinate frames
    T_all = robot.fkine_all(q_rad)
    scale = 0.08
    for T in T_all:
        origin = T.t
        ax.quiver(origin[0], origin[1], origin[2], 
                  T.R[0,0]*scale, T.R[1,0]*scale, T.R[2,0]*scale,
                  color='#ff6b6b', arrow_length_ratio=0.2, linewidth=1.5, alpha=0.7)
        ax.quiver(origin[0], origin[1], origin[2], 
                  T.R[0,1]*scale, T.R[1,1]*scale, T.R[2,1]*scale,
                  color='#4ecdc4', arrow_length_ratio=0.2, linewidth=1.5, alpha=0.7)
        ax.quiver(origin[0], origin[1], origin[2], 
                  T.R[0,2]*scale, T.R[1,2]*scale, T.R[2,2]*scale,
                  color='#74b9ff', arrow_length_ratio=0.2, linewidth=1.5, alpha=0.7)
    
    # World frame
    ax.quiver(0, 0, 0, 0.15, 0, 0, color='#ff6b6b', arrow_length_ratio=0.12, linewidth=2.5)
    ax.quiver(0, 0, 0, 0, 0.15, 0, color='#4ecdc4', arrow_length_ratio=0.12, linewidth=2.5)
    ax.quiver(0, 0, 0, 0, 0, 0.15, color='#74b9ff', arrow_length_ratio=0.12, linewidth=2.5)
    ax.text(0.18, 0, 0, 'X', color='#ff6b6b', fontsize=10, fontweight='bold')
    ax.text(0, 0.18, 0, 'Y', color='#4ecdc4', fontsize=10, fontweight='bold')
    ax.text(0, 0, 0.18, 'Z', color='#74b9ff', fontsize=10, fontweight='bold')
    
    # Axis settings
    ax.set_xlim([-max_reach, max_reach])
    ax.set_ylim([-max_reach, max_reach])
    ax.set_zlim([0, max_reach * 1.2])
    ax.set_xlabel('X (m)', fontsize=9, labelpad=6, color=COLORS['text_secondary'])
    ax.set_ylabel('Y (m)', fontsize=9, labelpad=6, color=COLORS['text_secondary'])
    ax.set_zlabel('Z (m)', fontsize=9, labelpad=6, color=COLORS['text_secondary'])
    ax.tick_params(colors=COLORS['text_secondary'], labelsize=7)
    ax.set_box_aspect([1, 1, 0.6])

def update_solution_display():
    global current_solution_idx
    
    if len(ik_solutions) == 0:
        text_solution.set_text('No solution found')
        text_sol_counter.set_text('')
        text_error.set_text('---')
        return
    
    if current_solution_idx >= len(ik_solutions):
        current_solution_idx = 0
    if current_solution_idx < 0:
        current_solution_idx = len(ik_solutions) - 1
    
    sol = ik_solutions[current_solution_idx]
    
    # Display solution
    text_solution.set_text(
        f'θ1 = {np.degrees(sol[0]):+7.2f}°\n'
        f'd2 = {sol[1]:+7.4f} m\n'
        f'θ3 = {np.degrees(sol[2]):+7.2f}°\n'
        f'θ4 = {np.degrees(sol[3]):+7.2f}°'
    )
    
    text_sol_counter.set_text(f'Solution {current_solution_idx + 1} of {len(ik_solutions)}')
    
    # Compute error
    T = robot.fkine(sol)
    error = np.linalg.norm(T.t - np.array(target_pos))
    text_error.set_text(f'{error*1000:.2f}\nmm')
    
    # Update robot
    draw_robot(sol, target_pos)
    fig.canvas.draw_idle()

def solve_ik(event):
    global ik_solutions, current_solution_idx, target_pos
    
    try:
        target_pos[0] = float(t_x.text)
        target_pos[1] = float(t_y.text)
        target_pos[2] = float(t_z.text)
    except:
        text_solution.set_text('Invalid target!')
        return
    
    # Try analytical IK first
    ik_solutions = ik_rprr(target_pos[0], target_pos[1], target_pos[2],
                           params['L0'], params['L1'], params['L2'], params['dmax'])
    
    # If no analytical solution, try numerical
    if len(ik_solutions) == 0:
        num_sol, error = ik_numerical(target_pos[0], target_pos[1], target_pos[2])
        if num_sol is not None and error < 0.05:
            ik_solutions = [num_sol.tolist()]
    
    current_solution_idx = 0
    update_solution_display()

def prev_solution(event):
    global current_solution_idx
    current_solution_idx -= 1
    update_solution_display()

def next_solution(event):
    global current_solution_idx
    current_solution_idx += 1
    update_solution_display()

def update_params(text):
    global robot
    try:
        params['L0'] = float(t_L0.text)
        params['L1'] = float(t_L1.text)
        params['L2'] = float(t_L2.text)
        params['dmax'] = float(t_dmax.text)
        robot = build_robot(params['L0'], params['L1'], params['L2'], params['dmax'])
        draw_robot(current_q, target_pos)
        fig.canvas.draw_idle()
    except:
        pass

# Connect events
btn_solve.on_clicked(solve_ik)
btn_prev.on_clicked(prev_solution)
btn_next.on_clicked(next_solution)

t_L0.on_submit(update_params)
t_L1.on_submit(update_params)
t_L2.on_submit(update_params)
t_dmax.on_submit(update_params)

# Initial draw
draw_robot(current_q, target_pos)

print("\n" + "="*55)
print("  RPRR Robot - Inverse Kinematics Simulator")
print("="*55)
print("  1. Enter target X, Y, Z coordinates")
print("  2. Click 'SOLVE IK' to compute joint values")
print("  3. Use PREV/NEXT to cycle through solutions")
print("="*55)

plt.show()
