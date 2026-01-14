import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons

# ============================================
# Color Palette - Modern Dark Theme
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
    'param': '#00d9a0',
    'workspace': '#4ecdc4',
}

# ============================================
# Initial Robot Parameters
# ============================================
params = {
    'L0': 0.5,
    'L1': 0.4,
    'L2': 0.3,
    'dmax': 0.5,
}

# Workspace visibility
show_workspace = False
workspace_points = None

# ============================================
# Robot Builder Function
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

def compute_workspace(robot, n_samples=12):
    """Compute workspace by sampling joint space"""
    points = []
    
    theta1_range = np.linspace(-pi, pi, n_samples)
    d2_range = np.linspace(0, params['dmax'], 4)
    theta3_range = np.linspace(-pi, pi, n_samples)
    theta4_range = np.linspace(-pi, pi, 6)
    
    for t1 in theta1_range:
        for d2 in d2_range:
            for t3 in theta3_range:
                for t4 in theta4_range:
                    q = [t1, d2, t3, t4]
                    T = robot.fkine(q)
                    points.append(T.t)
    
    return np.array(points)

robot = build_robot(params['L0'], params['L1'], params['L2'], params['dmax'])

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

q0_deg = [0.0, 0.2, 0.0, 0.0]
q0_rad = [np.radians(q0_deg[0]), q0_deg[1], np.radians(q0_deg[2]), np.radians(q0_deg[3])]

# ============================================
# Control Panel Background
# ============================================
panel_ax = fig.add_axes([0.58, 0.02, 0.40, 0.96])
panel_ax.set_facecolor(COLORS['bg_medium'])
panel_ax.set_xticks([])
panel_ax.set_yticks([])
for spine in panel_ax.spines.values():
    spine.set_color(COLORS['bg_light'])
    spine.set_linewidth(2)

fig.text(0.78, 0.96, 'RPRR ROBOT CONTROL', fontsize=16, fontweight='bold',
         ha='center', color=COLORS['text_primary'])

# ============================================
# Slider Styling Parameters
# ============================================
slider_height = 0.025
slider_width = 0.20
slider_left = 0.62
text_left = 0.83
text_width = 0.06

# ============================================
# JOINT CONTROLS Section
# ============================================
fig.text(0.78, 0.91, '──── JOINT CONTROLS ────', fontsize=9,
         ha='center', color=COLORS['text_secondary'])

joint_rows = [0.84, 0.76, 0.68, 0.60]
joint_labels = ['theta1  Rotation', 'd2  Extension', 'theta3  Rotation', 'theta4  Rotation']
joint_colors = [COLORS['link1'], COLORS['link2'], COLORS['link3'], COLORS['link4']]
joint_types = ['[R]', '[P]', '[R]', '[R]']

sliders = []
textboxes = []

for i, (row, label, color, jtype) in enumerate(zip(joint_rows, joint_labels, joint_colors, joint_types)):
    fig.text(slider_left - 0.02, row + 0.032, f'{jtype} {label}',
             fontsize=9, color=color, fontweight='bold', ha='left')
    
    ax_slider = plt.axes([slider_left, row, slider_width, slider_height])
    ax_slider.set_facecolor(COLORS['bg_light'])
    
    if i == 1:
        slider = Slider(ax_slider, '', 0.0, params['dmax'], valinit=q0_deg[i], 
                       color=color, valfmt='%.3f')
    else:
        slider = Slider(ax_slider, '', -180, 180, valinit=q0_deg[i], 
                       color=color, valfmt='%.0f', valstep=1)
    slider.valtext.set_visible(False)
    sliders.append(slider)
    
    ax_text = plt.axes([text_left, row, text_width, slider_height])
    ax_text.set_facecolor(COLORS['bg_light'])
    for spine in ax_text.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.5)
    
    if i == 1:
        textbox = TextBox(ax_text, '', initial=f'{q0_deg[i]:.3f}', 
                         color=COLORS['bg_light'], hovercolor=COLORS['bg_dark'])
    else:
        textbox = TextBox(ax_text, '', initial=f'{q0_deg[i]:.1f}',
                         color=COLORS['bg_light'], hovercolor=COLORS['bg_dark'])
    textboxes.append(textbox)
    
    unit = 'm' if i == 1 else 'deg'
    fig.text(text_left + text_width + 0.01, row + 0.01, unit, fontsize=8, color=COLORS['text_secondary'])

s_q1, s_q2, s_q3, s_q4 = sliders
t_q1, t_q2, t_q3, t_q4 = textboxes

# ============================================
# ROBOT PARAMETERS Section
# ============================================
fig.text(0.78, 0.52, '──── ROBOT PARAMETERS ────', fontsize=9,
         ha='center', color=COLORS['text_secondary'])

param_rows = [0.46, 0.40, 0.34, 0.28]
param_labels = ['L0   Base Height', 'L1   Link 1 Length', 'L2   End-Effector', 'dmax Max Extension']
param_keys = ['L0', 'L1', 'L2', 'dmax']
param_color = COLORS['param']

param_textboxes = []

for i, (row, label, key) in enumerate(zip(param_rows, param_labels, param_keys)):
    fig.text(slider_left - 0.02, row + 0.012, label,
             fontsize=10, color=param_color, fontweight='bold', ha='left')
    
    ax_text = plt.axes([0.76, row, 0.08, 0.03])
    ax_text.set_facecolor(COLORS['bg_dark'])
    for spine in ax_text.spines.values():
        spine.set_color(param_color)
        spine.set_linewidth(2)
    
    textbox = TextBox(ax_text, '', initial=f'{params[key]:.2f}', 
                     color=COLORS['bg_dark'], hovercolor=COLORS['bg_light'])
    param_textboxes.append(textbox)
    
    fig.text(0.85, row + 0.012, 'm', fontsize=10, color=COLORS['text_secondary'])

t_L0, t_L1, t_L2, t_dmax = param_textboxes

# ============================================
# WORKSPACE Toggle
# ============================================
ax_check = plt.axes([0.62, 0.21, 0.32, 0.05])
ax_check.set_facecolor(COLORS['bg_dark'])
for spine in ax_check.spines.values():
    spine.set_color(COLORS['workspace'])
    spine.set_linewidth(1.5)

check_workspace = CheckButtons(ax_check, ['Show Workspace'], [False])
check_workspace.labels[0].set_color(COLORS['workspace'])
check_workspace.labels[0].set_fontsize(11)
check_workspace.labels[0].set_fontweight('bold')

# ============================================
# End-Effector Display Panel
# ============================================
fig.text(0.78, 0.18, '──── END-EFFECTOR ────', fontsize=9,
         ha='center', color=COLORS['text_secondary'])

coord_height = 0.030
coord_width = 0.26
coord_left = 0.65

ax_x = plt.axes([coord_left, 0.115, coord_width, coord_height])
ax_y = plt.axes([coord_left, 0.08, coord_width, coord_height])
ax_z = plt.axes([coord_left, 0.045, coord_width, coord_height])

coord_axes = [ax_x, ax_y, ax_z]
coord_colors = ['#ff6b6b', '#4ecdc4', '#74b9ff']
coord_labels = ['X', 'Y', 'Z']

for ax_c, color, label in zip(coord_axes, coord_colors, coord_labels):
    ax_c.set_facecolor(COLORS['bg_dark'])
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
    ax_c.text(-0.08, 0.5, label, transform=ax_c.transAxes, fontsize=11,
              fontweight='bold', color=color, ha='center', va='center')

text_x = ax_x.text(0.5, 0.5, '+0.0000 m', ha='center', va='center', fontsize=13,
                   fontweight='bold', color='#ff6b6b', transform=ax_x.transAxes, family='monospace')
text_y = ax_y.text(0.5, 0.5, '+0.0000 m', ha='center', va='center', fontsize=13,
                   fontweight='bold', color='#4ecdc4', transform=ax_y.transAxes, family='monospace')
text_z = ax_z.text(0.5, 0.5, '+0.0000 m', ha='center', va='center', fontsize=13,
                   fontweight='bold', color='#74b9ff', transform=ax_z.transAxes, family='monospace')

# ============================================
# Reset Button
# ============================================
ax_btn = plt.axes([0.70, 0.008, 0.16, 0.03])
btn_reset = Button(ax_btn, 'RESET', color=COLORS['accent1'], hovercolor='#ff4757')
btn_reset.label.set_color('white')
btn_reset.label.set_fontweight('bold')
btn_reset.label.set_fontsize(10)

# ============================================
# Update Flag
# ============================================
updating = False

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

def draw_robot(q_rad):
    global workspace_points
    
    ax.cla()
    ax.set_facecolor(COLORS['bg_medium'])
    
    pos = get_link_positions(q_rad)
    max_reach = params['L0'] + params['dmax'] + params['L1'] + params['L2'] + 0.3
    
    # Draw workspace if enabled
    if show_workspace:
        if workspace_points is None:
            workspace_points = compute_workspace(robot, n_samples=15)
        
        # Draw as transparent point cloud
        ax.scatter3D(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2],
                    c=COLORS['workspace'], alpha=0.08, s=15, marker='o', edgecolors='none')
        
        # Draw convex hull outline
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(workspace_points)
            for simplex in hull.simplices:
                pts = workspace_points[simplex]
                ax.plot3D(pts[[0,1], 0], pts[[0,1], 1], pts[[0,1], 2], 
                         color=COLORS['workspace'], alpha=0.15, linewidth=0.5)
        except:
            pass
    
    # Floor grid
    floor_size = max_reach * 0.8
    for i in np.linspace(-floor_size, floor_size, 9):
        alpha = 0.3 - 0.15 * abs(i) / floor_size
        ax.plot3D([i, i], [-floor_size, floor_size], [0, 0], 
                  color=COLORS['grid'], alpha=alpha, linewidth=0.8)
        ax.plot3D([-floor_size, floor_size], [i, i], [0, 0], 
                  color=COLORS['grid'], alpha=alpha, linewidth=0.8)
    
    # Base platform
    theta_base = np.linspace(0, 2*pi, 40)
    r_base = 0.12
    x_base = r_base * np.cos(theta_base)
    y_base = r_base * np.sin(theta_base)
    ax.plot3D(x_base, y_base, np.zeros_like(x_base), 
              color=COLORS['accent1'], linewidth=4, alpha=0.8)
    
    # Draw links
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
    
    # Draw joints
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
    
    # Update coordinate display
    T_ee = robot.fkine(q_rad)
    text_x.set_text(f'{T_ee.t[0]:+.4f} m')
    text_y.set_text(f'{T_ee.t[1]:+.4f} m')
    text_z.set_text(f'{T_ee.t[2]:+.4f} m')

def get_current_q():
    return [np.radians(s_q1.val), s_q2.val, np.radians(s_q3.val), np.radians(s_q4.val)]

def toggle_workspace(label):
    global show_workspace, workspace_points
    show_workspace = not show_workspace
    if show_workspace:
        workspace_points = compute_workspace(robot, n_samples=15)
    draw_robot(get_current_q())
    fig.canvas.draw_idle()

def update_from_joint_sliders(val):
    global updating
    if updating:
        return
    updating = True
    
    t_q1.set_val(f'{s_q1.val:.1f}')
    t_q2.set_val(f'{s_q2.val:.3f}')
    t_q3.set_val(f'{s_q3.val:.1f}')
    t_q4.set_val(f'{s_q4.val:.1f}')
    
    draw_robot(get_current_q())
    fig.canvas.draw_idle()
    
    updating = False

def rebuild_robot():
    global robot, workspace_points
    robot = build_robot(params['L0'], params['L1'], params['L2'], params['dmax'])
    workspace_points = None
    
    s_q2.valmin = 0
    s_q2.valmax = params['dmax']
    s_q2.ax.set_xlim(0, params['dmax'])
    
    if s_q2.val > params['dmax']:
        s_q2.set_val(params['dmax'])
        t_q2.set_val(f'{params["dmax"]:.3f}')
    
    if show_workspace:
        workspace_points = compute_workspace(robot, n_samples=15)
    
    draw_robot(get_current_q())
    fig.canvas.draw_idle()

# Joint text box handlers
def update_q1_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = np.clip(float(text), -180, 180)
        s_q1.set_val(val)
        draw_robot(get_current_q())
        fig.canvas.draw_idle()
    except ValueError:
        pass
    updating = False

def update_q2_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = np.clip(float(text), 0, params['dmax'])
        s_q2.set_val(val)
        draw_robot(get_current_q())
        fig.canvas.draw_idle()
    except ValueError:
        pass
    updating = False

def update_q3_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = np.clip(float(text), -180, 180)
        s_q3.set_val(val)
        draw_robot(get_current_q())
        fig.canvas.draw_idle()
    except ValueError:
        pass
    updating = False

def update_q4_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = np.clip(float(text), -180, 180)
        s_q4.set_val(val)
        draw_robot(get_current_q())
        fig.canvas.draw_idle()
    except ValueError:
        pass
    updating = False

# Parameter text box handlers
def update_L0_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = float(text)
        if val > 0:
            params['L0'] = val
            rebuild_robot()
    except ValueError:
        pass
    updating = False

def update_L1_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = float(text)
        if val > 0:
            params['L1'] = val
            rebuild_robot()
    except ValueError:
        pass
    updating = False

def update_L2_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = float(text)
        if val > 0:
            params['L2'] = val
            rebuild_robot()
    except ValueError:
        pass
    updating = False

def update_dmax_text(text):
    global updating
    if updating:
        return
    updating = True
    try:
        val = float(text)
        if val > 0:
            params['dmax'] = val
            rebuild_robot()
    except ValueError:
        pass
    updating = False

def reset(event):
    global updating, robot, params, workspace_points
    updating = True
    
    params['L0'] = 0.5
    params['L1'] = 0.4
    params['L2'] = 0.3
    params['dmax'] = 0.5
    robot = build_robot(params['L0'], params['L1'], params['L2'], params['dmax'])
    workspace_points = None
    
    t_L0.set_val(f'{params["L0"]:.2f}')
    t_L1.set_val(f'{params["L1"]:.2f}')
    t_L2.set_val(f'{params["L2"]:.2f}')
    t_dmax.set_val(f'{params["dmax"]:.2f}')
    
    s_q2.valmax = params['dmax']
    s_q2.ax.set_xlim(0, params['dmax'])
    
    s_q1.set_val(q0_deg[0])
    s_q2.set_val(q0_deg[1])
    s_q3.set_val(q0_deg[2])
    s_q4.set_val(q0_deg[3])
    t_q1.set_val(f'{q0_deg[0]:.1f}')
    t_q2.set_val(f'{q0_deg[1]:.3f}')
    t_q3.set_val(f'{q0_deg[2]:.1f}')
    t_q4.set_val(f'{q0_deg[3]:.1f}')
    
    if show_workspace:
        workspace_points = compute_workspace(robot, n_samples=15)
    
    updating = False
    draw_robot(q0_rad)
    fig.canvas.draw_idle()

# Connect events
s_q1.on_changed(update_from_joint_sliders)
s_q2.on_changed(update_from_joint_sliders)
s_q3.on_changed(update_from_joint_sliders)
s_q4.on_changed(update_from_joint_sliders)

t_q1.on_submit(update_q1_text)
t_q2.on_submit(update_q2_text)
t_q3.on_submit(update_q3_text)
t_q4.on_submit(update_q4_text)

t_L0.on_submit(update_L0_text)
t_L1.on_submit(update_L1_text)
t_L2.on_submit(update_L2_text)
t_dmax.on_submit(update_dmax_text)

check_workspace.on_clicked(toggle_workspace)
btn_reset.on_clicked(reset)

draw_robot(q0_rad)

print("\n" + "="*50)
print("  RPRR Robot Simulator")
print("="*50)
print("  JOINT CONTROLS: Use sliders or type values")
print("  ROBOT PARAMETERS: Type value + press ENTER")
print("  WORKSPACE: Toggle checkbox to show/hide")
print("="*50)

plt.show()
