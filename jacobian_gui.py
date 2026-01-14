#!/usr/bin/env python3
"""
RPRR Robot Force/Torque Analysis GUI
Using Peter Corke's Robotics Toolbox for verified kinematics and Jacobian

"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
import warnings
warnings.filterwarnings('ignore')


class RPRRForceAnalyzer:
    def __init__(self):
        # Robot parameters (editable)
        self.L0 = 0.5  # Base height
        self.L1 = 0.4  # Link 1 length
        self.L2 = 0.3  # Tool length
        
        # Build robot
        self.build_robot()
        
        # Initial joint configuration
        self.theta1 = np.radians(30)
        self.d2 = 0.15
        self.theta3 = np.radians(45)
        self.theta4 = np.radians(30)
        
        # Applied end-effector force (N)
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = -10.0
        
        self.setup_gui()
        
    def build_robot(self):
        """Build robot with current parameters."""
        self.robot = rtb.DHRobot([
            RevoluteMDH(a=0, alpha=0, d=self.L0),
            PrismaticMDH(a=0, alpha=0, theta=0),
            RevoluteMDH(a=0, alpha=pi/2, d=0),
            RevoluteMDH(a=self.L1, alpha=0, d=0),
        ], tool=SE3.Tx(self.L2), name="RPRR")
        
    def get_joint_positions(self, q):
        """Get positions of all joints for visualization."""
        theta1, d2, theta3, theta4 = q
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c34, s34 = np.cos(theta3 + theta4), np.sin(theta3 + theta4)
        
        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.L0])
        p2 = np.array([0, 0, self.L0 + d2])
        p3 = np.array([0, 0, self.L0 + d2])
        p4 = np.array([c1 * self.L1 * c3, s1 * self.L1 * c3, self.L0 + d2 + self.L1 * s3])
        
        T = self.robot.fkine(q)
        p5 = T.t
        
        return p0, p1, p2, p3, p4, p5
    
    def setup_gui(self):
        """Setup the matplotlib GUI."""
        self.fig = plt.figure(figsize=(16, 11))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        self.fig.suptitle('RPRR Robot Force/Torque Analysis    τ = Jᵀ·F    (Robotics Toolbox)', 
                         fontsize=18, fontweight='bold', color='white', y=0.98)
        
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                              left=0.05, right=0.95, top=0.92, bottom=0.52,
                              wspace=0.20, hspace=0.25)
        
        self.ax_robot = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax_robot.set_facecolor('#16213e')
        
        self.ax_jacobian = self.fig.add_subplot(gs[0, 1])
        self.ax_jacobian.set_facecolor('#16213e')
        
        self.ax_torques = self.fig.add_subplot(gs[1, 1])
        self.ax_torques.set_facecolor('#16213e')
        
        self.setup_controls()
        self.update_plot()
        
    def setup_controls(self):
        """Setup sliders with text boxes and buttons."""
        slider_width = 0.12
        slider_height = 0.015
        textbox_width = 0.05
        textbox_height = 0.026
        gap = 0.008
        
        # ===== JOINT CONTROLS =====
        joint_x = 0.03
        joint_slider_x = joint_x + 0.055
        joint_text_x = joint_slider_x + slider_width + gap
        y_start = 0.34
        y_step = 0.050
        
        self.fig.text(joint_x + 0.09, y_start + 0.055, 'JOINTS', 
                     ha='center', fontsize=14, fontweight='bold', color='#e94560')
        
        # θ₁
        y = y_start
        self.fig.text(joint_x, y, 'θ₁ (°)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_t1 = self.fig.add_axes([joint_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_t1.set_facecolor('#0f3460')
        self.slider_t1 = Slider(ax_t1, '', -180, 180, valinit=np.degrees(self.theta1), color='#e94560')
        self.slider_t1.valtext.set_visible(False)
        ax_t1_text = self.fig.add_axes([joint_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_t1_text.set_facecolor('#e8e8e8')
        self.text_t1 = TextBox(ax_t1_text, '', initial=f'{np.degrees(self.theta1):.1f}', textalignment='center')
        self.text_t1.text_disp.set_color('#1a1a2e')
        self.text_t1.text_disp.set_fontsize(12)
        self.text_t1.text_disp.set_fontweight('bold')
        
        # d₂
        y -= y_step
        self.fig.text(joint_x, y, 'd₂ (m)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_d2 = self.fig.add_axes([joint_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_d2.set_facecolor('#0f3460')
        self.slider_d2 = Slider(ax_d2, '', 0, 0.5, valinit=self.d2, color='#e94560')
        self.slider_d2.valtext.set_visible(False)
        ax_d2_text = self.fig.add_axes([joint_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_d2_text.set_facecolor('#e8e8e8')
        self.text_d2 = TextBox(ax_d2_text, '', initial=f'{self.d2:.2f}', textalignment='center')
        self.text_d2.text_disp.set_color('#1a1a2e')
        self.text_d2.text_disp.set_fontsize(12)
        self.text_d2.text_disp.set_fontweight('bold')
        
        # θ₃
        y -= y_step
        self.fig.text(joint_x, y, 'θ₃ (°)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_t3 = self.fig.add_axes([joint_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_t3.set_facecolor('#0f3460')
        self.slider_t3 = Slider(ax_t3, '', -90, 90, valinit=np.degrees(self.theta3), color='#e94560')
        self.slider_t3.valtext.set_visible(False)
        ax_t3_text = self.fig.add_axes([joint_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_t3_text.set_facecolor('#e8e8e8')
        self.text_t3 = TextBox(ax_t3_text, '', initial=f'{np.degrees(self.theta3):.1f}', textalignment='center')
        self.text_t3.text_disp.set_color('#1a1a2e')
        self.text_t3.text_disp.set_fontsize(12)
        self.text_t3.text_disp.set_fontweight('bold')
        
        # θ₄
        y -= y_step
        self.fig.text(joint_x, y, 'θ₄ (°)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_t4 = self.fig.add_axes([joint_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_t4.set_facecolor('#0f3460')
        self.slider_t4 = Slider(ax_t4, '', -150, 150, valinit=np.degrees(self.theta4), color='#e94560')
        self.slider_t4.valtext.set_visible(False)
        ax_t4_text = self.fig.add_axes([joint_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_t4_text.set_facecolor('#e8e8e8')
        self.text_t4 = TextBox(ax_t4_text, '', initial=f'{np.degrees(self.theta4):.1f}', textalignment='center')
        self.text_t4.text_disp.set_color('#1a1a2e')
        self.text_t4.text_disp.set_fontsize(12)
        self.text_t4.text_disp.set_fontweight('bold')
        
        # ===== FORCE CONTROLS =====
        force_x = 0.28
        force_slider_x = force_x + 0.055
        force_text_x = force_slider_x + slider_width + gap
        y_start = 0.34
        
        self.fig.text(force_x + 0.09, y_start + 0.055, 'FORCES', 
                     ha='center', fontsize=14, fontweight='bold', color='#00ff88')
        
        # Fₓ
        y = y_start
        self.fig.text(force_x, y, 'Fₓ (N)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_fx = self.fig.add_axes([force_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_fx.set_facecolor('#0f3460')
        self.slider_fx = Slider(ax_fx, '', -20, 20, valinit=self.Fx, color='#00ff88')
        self.slider_fx.valtext.set_visible(False)
        ax_fx_text = self.fig.add_axes([force_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_fx_text.set_facecolor('#e8e8e8')
        self.text_fx = TextBox(ax_fx_text, '', initial=f'{self.Fx:.1f}', textalignment='center')
        self.text_fx.text_disp.set_color('#1a1a2e')
        self.text_fx.text_disp.set_fontsize(12)
        self.text_fx.text_disp.set_fontweight('bold')
        
        # Fᵧ
        y -= y_step
        self.fig.text(force_x, y, 'Fᵧ (N)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_fy = self.fig.add_axes([force_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_fy.set_facecolor('#0f3460')
        self.slider_fy = Slider(ax_fy, '', -20, 20, valinit=self.Fy, color='#00ff88')
        self.slider_fy.valtext.set_visible(False)
        ax_fy_text = self.fig.add_axes([force_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_fy_text.set_facecolor('#e8e8e8')
        self.text_fy = TextBox(ax_fy_text, '', initial=f'{self.Fy:.1f}', textalignment='center')
        self.text_fy.text_disp.set_color('#1a1a2e')
        self.text_fy.text_disp.set_fontsize(12)
        self.text_fy.text_disp.set_fontweight('bold')
        
        # Fᵤ
        y -= y_step
        self.fig.text(force_x, y, 'Fᵤ (N)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_fz = self.fig.add_axes([force_slider_x, y - slider_height/2, slider_width, slider_height])
        ax_fz.set_facecolor('#0f3460')
        self.slider_fz = Slider(ax_fz, '', -20, 20, valinit=self.Fz, color='#00ff88')
        self.slider_fz.valtext.set_visible(False)
        ax_fz_text = self.fig.add_axes([force_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_fz_text.set_facecolor('#e8e8e8')
        self.text_fz = TextBox(ax_fz_text, '', initial=f'{self.Fz:.1f}', textalignment='center')
        self.text_fz.text_disp.set_color('#1a1a2e')
        self.text_fz.text_disp.set_fontsize(12)
        self.text_fz.text_disp.set_fontweight('bold')
        
        # ===== ROBOT PARAMETERS =====
        param_x = 0.53
        param_text_x = param_x + 0.055
        y_start = 0.34
        
        self.fig.text(param_x + 0.045, y_start + 0.055, 'PARAMETERS', 
                     ha='center', fontsize=14, fontweight='bold', color='#ffd700')
        
        # L0
        y = y_start
        self.fig.text(param_x, y, 'L₀ (m)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_L0 = self.fig.add_axes([param_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_L0.set_facecolor('#e8e8e8')
        self.text_L0 = TextBox(ax_L0, '', initial=f'{self.L0:.2f}', textalignment='center')
        self.text_L0.text_disp.set_color('#1a1a2e')
        self.text_L0.text_disp.set_fontsize(12)
        self.text_L0.text_disp.set_fontweight('bold')
        
        # L1
        y -= y_step
        self.fig.text(param_x, y, 'L₁ (m)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_L1 = self.fig.add_axes([param_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_L1.set_facecolor('#e8e8e8')
        self.text_L1 = TextBox(ax_L1, '', initial=f'{self.L1:.2f}', textalignment='center')
        self.text_L1.text_disp.set_color('#1a1a2e')
        self.text_L1.text_disp.set_fontsize(12)
        self.text_L1.text_disp.set_fontweight('bold')
        
        # L2
        y -= y_step
        self.fig.text(param_x, y, 'L₂ (m)', color='white', fontsize=11, va='center', fontweight='bold')
        ax_L2 = self.fig.add_axes([param_text_x, y - textbox_height/2, textbox_width, textbox_height])
        ax_L2.set_facecolor('#e8e8e8')
        self.text_L2 = TextBox(ax_L2, '', initial=f'{self.L2:.2f}', textalignment='center')
        self.text_L2.text_disp.set_color('#1a1a2e')
        self.text_L2.text_disp.set_fontsize(12)
        self.text_L2.text_disp.set_fontweight('bold')
        
        # ===== HOME BUTTON =====
        btn_x = 0.75
        btn_width = 0.10
        btn_height = 0.045
        btn_y = 0.34
        
        ax_home = self.fig.add_axes([btn_x, btn_y, btn_width, btn_height])
        self.btn_home = Button(ax_home, 'Home', color='#0f3460', hovercolor='#e94560')
        self.btn_home.label.set_color('white')
        self.btn_home.label.set_fontsize(12)
        self.btn_home.label.set_fontweight('bold')
        
        # Connect callbacks
        self.slider_t1.on_changed(lambda val: self.on_slider_change('t1', val))
        self.slider_d2.on_changed(lambda val: self.on_slider_change('d2', val))
        self.slider_t3.on_changed(lambda val: self.on_slider_change('t3', val))
        self.slider_t4.on_changed(lambda val: self.on_slider_change('t4', val))
        self.slider_fx.on_changed(lambda val: self.on_slider_change('fx', val))
        self.slider_fy.on_changed(lambda val: self.on_slider_change('fy', val))
        self.slider_fz.on_changed(lambda val: self.on_slider_change('fz', val))
        
        self.text_t1.on_submit(lambda val: self.on_text_submit('t1', val))
        self.text_d2.on_submit(lambda val: self.on_text_submit('d2', val))
        self.text_t3.on_submit(lambda val: self.on_text_submit('t3', val))
        self.text_t4.on_submit(lambda val: self.on_text_submit('t4', val))
        self.text_fx.on_submit(lambda val: self.on_text_submit('fx', val))
        self.text_fy.on_submit(lambda val: self.on_text_submit('fy', val))
        self.text_fz.on_submit(lambda val: self.on_text_submit('fz', val))
        
        self.text_L0.on_submit(lambda val: self.on_param_change('L0', val))
        self.text_L1.on_submit(lambda val: self.on_param_change('L1', val))
        self.text_L2.on_submit(lambda val: self.on_param_change('L2', val))
        
        self.btn_home.on_clicked(self.preset_home)
        
    def on_slider_change(self, name, val):
        if name == 't1':
            self.theta1 = np.radians(val)
            self.text_t1.set_val(f'{val:.1f}')
        elif name == 'd2':
            self.d2 = val
            self.text_d2.set_val(f'{val:.2f}')
        elif name == 't3':
            self.theta3 = np.radians(val)
            self.text_t3.set_val(f'{val:.1f}')
        elif name == 't4':
            self.theta4 = np.radians(val)
            self.text_t4.set_val(f'{val:.1f}')
        elif name == 'fx':
            self.Fx = val
            self.text_fx.set_val(f'{val:.1f}')
        elif name == 'fy':
            self.Fy = val
            self.text_fy.set_val(f'{val:.1f}')
        elif name == 'fz':
            self.Fz = val
            self.text_fz.set_val(f'{val:.1f}')
        self.update_plot()
        
    def on_text_submit(self, name, val):
        try:
            v = float(val)
            if name == 't1':
                v = np.clip(v, -180, 180)
                self.theta1 = np.radians(v)
                self.slider_t1.set_val(v)
            elif name == 'd2':
                v = np.clip(v, 0, 0.5)
                self.d2 = v
                self.slider_d2.set_val(v)
            elif name == 't3':
                v = np.clip(v, -90, 90)
                self.theta3 = np.radians(v)
                self.slider_t3.set_val(v)
            elif name == 't4':
                v = np.clip(v, -150, 150)
                self.theta4 = np.radians(v)
                self.slider_t4.set_val(v)
            elif name == 'fx':
                v = np.clip(v, -20, 20)
                self.Fx = v
                self.slider_fx.set_val(v)
            elif name == 'fy':
                v = np.clip(v, -20, 20)
                self.Fy = v
                self.slider_fy.set_val(v)
            elif name == 'fz':
                v = np.clip(v, -20, 20)
                self.Fz = v
                self.slider_fz.set_val(v)
            self.update_plot()
        except ValueError:
            pass
            
    def on_param_change(self, name, val):
        """Handle robot parameter changes - rebuilds robot."""
        try:
            v = float(val)
            if v <= 0:
                return  # Must be positive
            if name == 'L0':
                self.L0 = v
            elif name == 'L1':
                self.L1 = v
            elif name == 'L2':
                self.L2 = v
            self.build_robot()
            self.update_plot()
        except ValueError:
            pass
        
    def preset_home(self, event):
        self.slider_t1.set_val(0)
        self.slider_d2.set_val(0.1)
        self.slider_t3.set_val(0)
        self.slider_t4.set_val(0)
        self.slider_fx.set_val(0)
        self.slider_fy.set_val(0)
        self.slider_fz.set_val(-10)
        
    def update_plot(self):
        """Update all plots using robotics toolbox."""
        q = [self.theta1, self.d2, self.theta3, self.theta4]
        
        T = self.robot.fkine(q)
        p5 = T.t
        
        J = self.robot.jacob0(q)
        Jv = J[:3, :]
        
        p0, p1, p2, p3, p4, p5 = self.get_joint_positions(q)
        
        F = np.array([self.Fx, self.Fy, self.Fz])
        tau = Jv.T @ F
        F_norm = np.linalg.norm(F)
        
        rank = np.linalg.matrix_rank(Jv)
        is_singular = rank < 3
        
        try:
            w = np.sqrt(np.linalg.det(Jv @ Jv.T))
        except:
            w = 0.0
        
        # ====== 3D Robot Plot ======
        self.ax_robot.clear()
        self.ax_robot.set_facecolor('#16213e')
        
        theta_base = np.linspace(0, 2*np.pi, 30)
        r_base = 0.12
        x_base = r_base * np.cos(theta_base)
        y_base = r_base * np.sin(theta_base)
        self.ax_robot.plot(x_base, y_base, np.zeros_like(theta_base), color='gray', linewidth=2)
        self.ax_robot.plot_surface(
            np.outer(np.linspace(0, r_base, 2), np.cos(theta_base)),
            np.outer(np.linspace(0, r_base, 2), np.sin(theta_base)),
            np.zeros((2, len(theta_base))),
            color='gray', alpha=0.4
        )
        
        self.ax_robot.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 
                          color='#4a90d9', linewidth=10, solid_capstyle='round')
        self.ax_robot.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                          color='#00bcd4', linewidth=12, solid_capstyle='round')
        self.ax_robot.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 
                          color='#e94560', linewidth=8, solid_capstyle='round')
        self.ax_robot.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]], 
                          color='#ff9800', linewidth=6, solid_capstyle='round')
        
        for pos, color in zip([p1, p2, p3, p4], ['#4a90d9', '#00bcd4', '#e94560', '#ff9800']):
            self.ax_robot.scatter(*pos, color=color, s=120, edgecolors='white', linewidths=2, zorder=5)
        
        self.ax_robot.scatter(*p5, color='#00ff88', s=200, marker='*', edgecolors='white', linewidths=1.5, zorder=6)
        
        if F_norm > 0.5:
            scale = 0.015
            self.ax_robot.quiver(p5[0], p5[1], p5[2], F[0]*scale, F[1]*scale, F[2]*scale,
                                color='yellow', arrow_length_ratio=0.15, linewidth=3, zorder=10)
        
        if is_singular:
            self.ax_robot.text2D(0.5, 0.95, '⚠ SINGULARITY ⚠', transform=self.ax_robot.transAxes,
                                fontsize=14, color='red', fontweight='bold', ha='center',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
        
        # Dynamic axis limits based on robot size
        max_reach = self.L0 + 0.5 + self.L1 + self.L2 + 0.2
        self.ax_robot.set_xlim([-max_reach, max_reach])
        self.ax_robot.set_ylim([-max_reach, max_reach])
        self.ax_robot.set_zlim([0, max_reach])
        self.ax_robot.set_xlabel('X', color='white', fontsize=11)
        self.ax_robot.set_ylabel('Y', color='white', fontsize=11)
        self.ax_robot.set_zlabel('Z', color='white', fontsize=11)
        self.ax_robot.set_title(f'EE: ({p5[0]:.3f}, {p5[1]:.3f}, {p5[2]:.3f})   Rank={rank}   w={w:.4f}', 
                               color='white', fontsize=12)
        self.ax_robot.tick_params(colors='white', labelsize=9)
        self.ax_robot.xaxis.pane.fill = False
        self.ax_robot.yaxis.pane.fill = False
        self.ax_robot.zaxis.pane.fill = False
        
        # ====== Jacobian Matrix ======
        self.ax_jacobian.clear()
        self.ax_jacobian.set_facecolor('#16213e')
        self.ax_jacobian.axis('off')
        self.ax_jacobian.set_title('Jacobian Jᵥ (from RTB)', color='white', fontsize=14, fontweight='bold')
        
        cell_text = [[f'{Jv[i,j]:7.3f}' for j in range(4)] for i in range(3)]
        col_labels = ['∂/∂θ₁', '∂/∂d₂', '∂/∂θ₃', '∂/∂θ₄']
        row_labels = ['vₓ', 'vᵧ', 'vᵤ']
        
        table = self.ax_jacobian.table(cellText=cell_text, colLabels=col_labels,
                                       rowLabels=row_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.2)
        
        # Color cells - black text
        max_val = np.max(np.abs(Jv)) + 0.01
        for i in range(3):
            for j in range(4):
                cell = table[(i+1, j)]
                val = Jv[i,j] / max_val
                cell.set_facecolor(plt.cm.RdBu_r(0.5 + 0.4 * val))
                cell.set_text_props(color='black', fontweight='bold')
        for j in range(4):
            table[(0, j)].set_facecolor('#0f3460')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(3):
            table[(i+1, -1)].set_facecolor('#0f3460')
            table[(i+1, -1)].set_text_props(color='white', fontweight='bold')
        
        # ====== Joint Torques ======
        self.ax_torques.clear()
        self.ax_torques.set_facecolor('#16213e')
        
        joint_names = ['τ₁ (Nm)', 'F₂ (N)', 'τ₃ (Nm)', 'τ₄ (Nm)']
        colors = ['#e94560' if t < 0 else '#00ff88' for t in tau]
        
        bars = self.ax_torques.barh(joint_names, tau, color=colors, edgecolor='white', linewidth=1.5, height=0.6)
        self.ax_torques.axvline(x=0, color='white', linewidth=1)
        
        for bar, val in zip(bars, tau):
            x_pos = val + (0.5 if val >= 0 else -0.5)
            self.ax_torques.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.2f}',
                                va='center', ha='left' if val >= 0 else 'right',
                                color='white', fontsize=11, fontweight='bold')
        
        self.ax_torques.set_xlabel('Torque / Force', color='white', fontsize=11)
        self.ax_torques.set_title('Joint Torques  τ = Jᵀ·F', color='white', fontsize=14, fontweight='bold')
        self.ax_torques.tick_params(colors='white', labelsize=10)
        max_tau = max(abs(min(tau)), abs(max(tau)), 5)
        self.ax_torques.set_xlim([-max_tau - 2, max_tau + 2])
        self.ax_torques.grid(True, alpha=0.3, axis='x', color='white')
        for spine in self.ax_torques.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        self.fig.canvas.draw_idle()
        
    def run(self):
        plt.show()


def main():
    print("=" * 60)
    print("   RPRR Robot Force/Torque Analysis GUI")
    print("   Using Peter Corke's Robotics Toolbox")
    print("=" * 60)
    print("\n  Formula: τ = Jᵀ · F")
    print("\n  • Sliders or text input (Enter to apply)")
    print("  • Change L0, L1, L2 to modify robot geometry")
    print("=" * 60)
    
    analyzer = RPRRForceAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
