#!/usr/bin/env python3
"""
RPRR Robot Trajectory Animation
Trajectories that exercise all joints:
- Straight Line (diagonal in 3D)
- Circular Arc (in XY plane with height change)
- Sinusoidal (wave with rotation)

"""

import numpy as np
from math import pi, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
import warnings
warnings.filterwarnings('ignore')


class TrajectoryAnimator:
    def __init__(self):
        # Robot parameters
        self.L0 = 0.5
        self.L1 = 0.4
        self.L2 = 0.3
        
        # Build robot
        self.build_robot()
        
        # Trajectory parameters
        self.T = 5.0  # Duration (seconds)
        self.n_points = 200
        
        # Animation state
        self.current_idx = 0
        self.playing = False
        self.anim = None
        
        # Trajectory type
        self.traj_type = 'line'
        
        # Compute initial trajectory
        self.compute_trajectory()
        
        self.setup_gui()
    
    def build_robot(self):
        """Build RPRR robot."""
        self.robot = rtb.DHRobot([
            RevoluteMDH(a=0, alpha=0, d=self.L0),
            PrismaticMDH(a=0, alpha=0, theta=0),
            RevoluteMDH(a=0, alpha=pi/2, d=0),
            RevoluteMDH(a=self.L1, alpha=0, d=0),
        ], tool=SE3.Tx(self.L2), name="RPRR")
    
    def cubic_time_scaling(self, t, T):
        """Cubic polynomial time scaling s(t) from 0 to 1."""
        tau = t / T
        s = 3*tau**2 - 2*tau**3
        return s
    
    def inverse_kinematics(self, p_target):
        """IK for the RPRR robot."""
        x, y, z = p_target
        
        theta1 = atan2(y, x) if (abs(x) > 0.001 or abs(y) > 0.001) else 0
        R = sqrt(x**2 + y**2)
        L1, L2 = self.L1, self.L2
        
        d2_est = 0.15
        z_arm = z - self.L0 - d2_est
        
        D = (R**2 + z_arm**2 - L1**2 - L2**2) / (2*L1*L2)
        D = np.clip(D, -1, 1)
        
        theta4 = atan2(-sqrt(max(0, 1 - D**2)), D)
        
        alpha = atan2(z_arm, R) if R > 0.001 else pi/2
        beta = atan2(L2*sin(theta4), L1 + L2*cos(theta4))
        theta3 = alpha - beta
        
        d2 = z - self.L0 - L1*sin(theta3) - L2*sin(theta3 + theta4)
        d2 = np.clip(d2, 0.05, 0.35)
        
        return np.array([theta1, d2, theta3, theta4])
    
    def compute_trajectory(self):
        """Compute trajectory points - all trajectories use all joints."""
        self.t_array = np.linspace(0, self.T, self.n_points)
        self.p_traj = np.zeros((self.n_points, 3))
        self.q_traj = np.zeros((self.n_points, 4))
        
        if self.traj_type == 'line':
            # Diagonal line using all joints
            # Start: front-right, low, arm extended down
            # End: front-left, high, arm folded up
            q_start = np.array([np.radians(-30), 0.08, np.radians(20), np.radians(-40)])
            q_end = np.array([np.radians(60), 0.30, np.radians(60), np.radians(-80)])
            
            for i, t in enumerate(self.t_array):
                s = self.cubic_time_scaling(t, self.T)
                self.q_traj[i] = q_start + s * (q_end - q_start)
                # Compute FK for Cartesian path
                T_mat = self.robot.fkine(self.q_traj[i])
                self.p_traj[i] = T_mat.t
        
        elif self.traj_type == 'circle':
            # Circle: θ1 rotates, d2 goes up/down, arm adjusts
            q_start = np.array([np.radians(-60), 0.10, np.radians(30), np.radians(-60)])
            q_end = np.array([np.radians(120), 0.28, np.radians(50), np.radians(-90)])
            
            for i, t in enumerate(self.t_array):
                s = self.cubic_time_scaling(t, self.T)
                # θ1 does full rotation
                theta1 = q_start[0] + s * (q_end[0] - q_start[0])
                # d2 goes up then down (sinusoidal)
                d2 = 0.18 + 0.12 * sin(pi * s)
                # θ3, θ4 change smoothly
                theta3 = q_start[2] + s * (q_end[2] - q_start[2])
                theta4 = q_start[3] + s * (q_end[3] - q_start[3])
                
                self.q_traj[i] = np.array([theta1, d2, theta3, theta4])
                T_mat = self.robot.fkine(self.q_traj[i])
                self.p_traj[i] = T_mat.t
        
        elif self.traj_type == 'sine':
            # Sinusoidal: all joints oscillate
            q_center = np.array([np.radians(15), 0.20, np.radians(45), np.radians(-70)])
            
            for i, t in enumerate(self.t_array):
                s = self.cubic_time_scaling(t, self.T)
                
                # θ1 sweeps with small oscillation
                theta1 = np.radians(-45) + s * np.radians(105) + np.radians(10) * sin(4*pi*s)
                # d2 oscillates up and down (2 cycles)
                d2 = 0.20 + 0.12 * sin(2 * 2*pi * s)
                # θ3 changes with wave
                theta3 = np.radians(30) + s * np.radians(30) + np.radians(15) * sin(2*pi*s)
                # θ4 counter-oscillates
                theta4 = np.radians(-50) - s * np.radians(30) - np.radians(20) * sin(2*pi*s)
                
                self.q_traj[i] = np.array([theta1, np.clip(d2, 0.08, 0.35), theta3, theta4])
                T_mat = self.robot.fkine(self.q_traj[i])
                self.p_traj[i] = T_mat.t
    
    def get_joint_positions(self, q):
        """Get positions of all joints."""
        theta1, d2, theta3, theta4 = q
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c34 = np.cos(theta3 + theta4)
        s34 = np.sin(theta3 + theta4)
        
        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.L0])
        p2 = np.array([0, 0, self.L0 + d2])
        p3 = p2.copy()
        p4 = np.array([c1 * self.L1 * c3, s1 * self.L1 * c3, self.L0 + d2 + self.L1 * s3])
        p5 = np.array([c1 * (self.L1 * c3 + self.L2 * c34), 
                       s1 * (self.L1 * c3 + self.L2 * c34), 
                       self.L0 + d2 + self.L1 * s3 + self.L2 * s34])
        
        return p0, p1, p2, p3, p4, p5
    
    def setup_gui(self):
        """Setup the matplotlib GUI."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # 3D Robot view
        self.ax = self.fig.add_axes([0.02, 0.12, 0.60, 0.82], projection='3d')
        self.ax.set_facecolor('#16213e')
        
        # Title
        self.fig.suptitle('RPRR Robot Trajectory Animation', 
                         fontsize=18, fontweight='bold', color='white', y=0.97)
        
        # Trajectory type radio buttons
        ax_radio = self.fig.add_axes([0.65, 0.70, 0.15, 0.20])
        ax_radio.set_facecolor('#16213e')
        ax_radio.set_title('Trajectory Type', color='white', fontsize=11, fontweight='bold', pad=10)
        self.radio = RadioButtons(ax_radio, ('Line', 'Circle', 'Sinusoidal'), activecolor='#e94560')
        for label in self.radio.labels:
            label.set_color('white')
            label.set_fontsize(11)
        self.radio.on_clicked(self.on_traj_type_change)
        
        # Play button
        ax_play = self.fig.add_axes([0.65, 0.55, 0.15, 0.08])
        self.btn_play = Button(ax_play, '▶  Play', color='#0f3460', hovercolor='#00ff88')
        self.btn_play.label.set_color('white')
        self.btn_play.label.set_fontsize(14)
        self.btn_play.on_clicked(self.toggle_play)
        
        # Reset button
        ax_reset = self.fig.add_axes([0.65, 0.46, 0.15, 0.06])
        self.btn_reset = Button(ax_reset, 'Reset', color='#0f3460', hovercolor='#e94560')
        self.btn_reset.label.set_color('white')
        self.btn_reset.label.set_fontsize(11)
        self.btn_reset.on_clicked(self.reset_animation)
        
        # Joint values display area
        self.ax_joints = self.fig.add_axes([0.82, 0.45, 0.16, 0.47])
        self.ax_joints.set_facecolor('#16213e')
        self.ax_joints.axis('off')
        
        # Info text (end-effector)
        self.info_text = self.fig.text(0.65, 0.38, '', fontsize=10, color='white',
                                       family='monospace', verticalalignment='top')
        

        
        self.update_plot()
    
    def on_traj_type_change(self, label):
        type_map = {'Line': 'line', 'Circle': 'circle', 'Sinusoidal': 'sine'}
        self.traj_type = type_map[label]
        self.current_idx = 0
        self.compute_trajectory()
        self.update_plot()
    
    def toggle_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text('⏸  Pause')
            self.anim = FuncAnimation(
                self.fig, 
                self.animate_frame,
                interval=25,
                blit=False,
                cache_frame_data=False
            )
            plt.draw()
        else:
            self.btn_play.label.set_text('▶  Play')
            if self.anim is not None:
                self.anim.event_source.stop()
    
    def animate_frame(self, frame):
        """Animation frame update function."""
        if not self.playing:
            return []
        
        self.current_idx += 1
        if self.current_idx >= self.n_points:
            self.current_idx = 0
        
        self.update_plot()
        return []
    
    def reset_animation(self, event):
        self.playing = False
        if self.anim is not None:
            self.anim.event_source.stop()
        self.current_idx = 0
        self.btn_play.label.set_text('▶  Play')
        self.update_plot()
    
    def update_plot(self):
        """Update all plots."""
        idx = self.current_idx
        q_current = self.q_traj[idx]
        p_current = self.p_traj[idx]
        
        # Get start and end joint values for display
        q_start = self.q_traj[0]
        q_end = self.q_traj[-1]
        
        # ===== 3D Robot Plot =====
        self.ax.clear()
        self.ax.set_facecolor('#16213e')
        
        # Draw coordinate axes at origin
        axis_length = 0.25
        # X axis (red)
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='#ff4444', arrow_length_ratio=0.15, linewidth=3)
        self.ax.text(axis_length + 0.05, 0, 0, 'X', color='#ff4444', fontsize=12, fontweight='bold')
        # Y axis (green)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='#44ff44', arrow_length_ratio=0.15, linewidth=3)
        self.ax.text(0, axis_length + 0.05, 0, 'Y', color='#44ff44', fontsize=12, fontweight='bold')
        # Z axis (blue)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='#4444ff', arrow_length_ratio=0.15, linewidth=3)
        self.ax.text(0, 0, axis_length + 0.05, 'Z', color='#4444ff', fontsize=12, fontweight='bold')
        
        # Draw ground plane grid
        grid_size = 0.6
        grid_lines = 7
        for i in np.linspace(-grid_size, grid_size, grid_lines):
            self.ax.plot([i, i], [-grid_size, grid_size], [0, 0], color='#333333', linewidth=0.5, alpha=0.5)
            self.ax.plot([-grid_size, grid_size], [i, i], [0, 0], color='#333333', linewidth=0.5, alpha=0.5)
        
        # Path styling
        if self.traj_type == 'line':
            path_color = '#00bcd4'
            path_label = 'Straight Line'
        elif self.traj_type == 'circle':
            path_color = '#ff9800'
            path_label = 'Circular Arc'
        else:
            path_color = '#e91e63'
            path_label = 'Sinusoidal'
        
        # Full trajectory path (dim)
        self.ax.plot(self.p_traj[:, 0], self.p_traj[:, 1], self.p_traj[:, 2],
                    color=path_color, linewidth=2, alpha=0.3, linestyle='--', label=path_label)
        
        # Traversed path (bright)
        if idx > 0:
            self.ax.plot(self.p_traj[:idx+1, 0], self.p_traj[:idx+1, 1], self.p_traj[:idx+1, 2],
                        color='#00ff88', linewidth=4, alpha=0.9)
        
        # Get joint positions
        p0, p1, p2, p3, p4, p5 = self.get_joint_positions(q_current)
        
        # Base platform
        theta_base = np.linspace(0, 2*np.pi, 30)
        r_base = 0.12
        x_base = r_base * np.cos(theta_base)
        y_base = r_base * np.sin(theta_base)
        self.ax.plot(x_base, y_base, np.zeros_like(theta_base), color='#555555', linewidth=3)
        
        # Robot links
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 
                    color='#4a90d9', linewidth=12, solid_capstyle='round')
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='#00bcd4', linewidth=14, solid_capstyle='round')
        self.ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 
                    color='#e94560', linewidth=9, solid_capstyle='round')
        self.ax.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]], 
                    color='#ff9800', linewidth=7, solid_capstyle='round')
        
        # Joints
        for pos, color, size in zip([p0, p1, p2, p4], 
                                    ['#4a90d9', '#00bcd4', '#00bcd4', '#e94560'],
                                    [100, 120, 120, 90]):
            self.ax.scatter(*pos, color=color, s=size, edgecolors='white', linewidths=2, zorder=5)
        
        # End effector
        self.ax.scatter(*p5, color='#00ff88', s=200, marker='*', edgecolors='white', linewidths=2, zorder=6)
        
        # Start/End markers
        self.ax.scatter(*self.p_traj[0], color='lime', s=100, marker='o', edgecolors='white', linewidths=2, label='Start')
        self.ax.scatter(*self.p_traj[-1], color='red', s=100, marker='s', edgecolors='white', linewidths=2, label='End')
        
        # Axis settings
        self.ax.set_xlim([-0.6, 0.8])
        self.ax.set_ylim([-0.6, 0.8])
        self.ax.set_zlim([0, 1.4])
        self.ax.set_xlabel('X (m)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (m)', color='white', fontsize=10)
        self.ax.set_zlabel('Z (m)', color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.legend(loc='upper left', fontsize=8, facecolor='#16213e', edgecolor='white', labelcolor='white')
        
        # ===== Joint Values Display =====
        self.ax_joints.clear()
        self.ax_joints.set_facecolor('#16213e')
        self.ax_joints.axis('off')
        
        # Title
        self.ax_joints.text(0.5, 0.98, 'JOINT VALUES', transform=self.ax_joints.transAxes,
                          fontsize=12, fontweight='bold', color='#e94560',
                          ha='center', va='top')
        
        # Table header
        self.ax_joints.text(0.5, 0.88, '─' * 24, transform=self.ax_joints.transAxes,
                          fontsize=10, color='white', ha='center', family='monospace')
        
        # Joint names and colors
        joint_info = [
            ('θ₁', np.degrees(q_current[0]), '°', np.degrees(q_start[0]), np.degrees(q_end[0]), '#e94560'),
            ('d₂', q_current[1] * 100, 'cm', q_start[1] * 100, q_end[1] * 100, '#00ff88'),
            ('θ₃', np.degrees(q_current[2]), '°', np.degrees(q_start[2]), np.degrees(q_end[2]), '#ffd700'),
            ('θ₄', np.degrees(q_current[3]), '°', np.degrees(q_start[3]), np.degrees(q_end[3]), '#00bcd4'),
        ]
        
        y_pos = 0.80
        for name, val, unit, start_val, end_val, color in joint_info:
            # Joint name
            self.ax_joints.text(0.05, y_pos, f'{name}:', transform=self.ax_joints.transAxes,
                              fontsize=14, fontweight='bold', color=color, family='monospace')
            # Current value (large)
            self.ax_joints.text(0.95, y_pos, f'{val:+7.1f}{unit}', transform=self.ax_joints.transAxes,
                              fontsize=14, fontweight='bold', color='white', ha='right', family='monospace')
            # Start → End range
            self.ax_joints.text(0.5, y_pos - 0.06, f'({start_val:+.1f} → {end_val:+.1f})',
                              transform=self.ax_joints.transAxes,
                              fontsize=9, color='#888888', ha='center', family='monospace')
            y_pos -= 0.18
        
        # ===== End-Effector Info =====
        t_current = self.t_array[idx]
        progress = (idx / (self.n_points - 1)) * 100
        
        info = (f"Time: {t_current:.2f} / {self.T:.1f} s\n"
                f"Progress: {progress:.0f}%\n\n"
                f"End-Effector (m):\n"
                f"  X: {p_current[0]:+.3f}\n"
                f"  Y: {p_current[1]:+.3f}\n"
                f"  Z: {p_current[2]:+.3f}")
        self.info_text.set_text(info)
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        plt.show()


def main():
    print("=" * 65)
    print("   RPRR Robot Trajectory Animation")
    print("=" * 65)
    print("\n  Trajectories (all use every joint):")
    print("  • Line       - Diagonal path: rotation + height + reach")
    print("  • Circle     - Arc in XY plane with Z variation")
    print("  • Sinusoidal - Wave with rotation sweep")
    print("\n  Controls:")
    print("  • Play/Pause - Start/stop animation")
    print("  • Reset      - Return to start position")
    print("=" * 65)
    
    animator = TrajectoryAnimator()
    animator.run()


if __name__ == "__main__":
    main()
