#!/usr/bin/env python3
"""
RPRR Robot Dynamics Simulator GUI
τ = M(q)q̈ + C(q,q̇)q̇ + g(q)

Interactive visualization of manipulator dynamics equation

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


class RPRRDynamicsSimulator:
    def __init__(self):
        # Robot parameters
        self.L0 = 0.5  # Base height
        self.L1 = 0.4  # Link 1 length
        self.L2 = 0.3  # Tool length
        
        # Mass properties
        self.m1 = 2.0  # Prismatic slider mass
        self.m2 = 1.5  # Upper arm mass
        self.m3 = 0.8  # Forearm + tool mass
        
        # Inertias
        self.I2 = (1/12) * self.m2 * self.L1**2
        self.I3 = (1/12) * self.m3 * self.L2**2
        
        # Gravity
        self.g = 9.81
        
        # Flag to prevent recursive updates
        self._updating = False
        
        # Build robot
        self.build_robot()
        
        # Initial state: position, velocity, acceleration
        self.q = np.array([np.radians(30), 0.15, np.radians(45), np.radians(30)])
        self.qd = np.array([0.0, 0.0, 0.0, 0.0])
        self.qdd = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.setup_gui()
        
    def build_robot(self):
        """Build RPRR robot with full dynamic properties."""
        lc2 = self.L1 / 2
        lc3 = self.L2 / 2
        
        link1 = RevoluteMDH(a=0, alpha=0, d=self.L0)
        link1.m = 0
        link1.r = np.array([0, 0, 0])
        link1.I = np.zeros((3, 3))
        
        link2 = PrismaticMDH(a=0, alpha=0, theta=0)
        link2.m = self.m1
        link2.r = np.array([0, 0, 0])
        link2.I = np.zeros((3, 3))
        
        link3 = RevoluteMDH(a=0, alpha=pi/2, d=0)
        link3.m = self.m2
        link3.r = np.array([lc2, 0, 0])
        link3.I = np.array([[0, 0, 0], [0, self.I2, 0], [0, 0, self.I2]])
        
        link4 = RevoluteMDH(a=self.L1, alpha=0, d=0)
        link4.m = self.m3
        link4.r = np.array([lc3, 0, 0])
        link4.I = np.array([[0, 0, 0], [0, self.I3, 0], [0, 0, self.I3]])
        
        self.robot = rtb.DHRobot([link1, link2, link3, link4],
                                  tool=SE3.Tx(self.L2),
                                  name="RPRR",
                                  gravity=[0, 0, -self.g])
    
    def get_joint_positions(self, q):
        """Get positions of all joints for visualization."""
        theta1, d2, theta3, theta4 = q
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        
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
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        self.fig.suptitle('RPRR Robot Dynamics Simulator    τ = M(q)q̈ + C(q,q̇)q̇ + g(q)', 
                         fontsize=16, fontweight='bold', color='white', y=0.98)
        
        # Grid layout
        gs = gridspec.GridSpec(3, 4, figure=self.fig,
                              left=0.03, right=0.97, top=0.93, bottom=0.42,
                              wspace=0.25, hspace=0.35)
        
        # 3D Robot view
        self.ax_robot = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self.ax_robot.set_facecolor('#16213e')
        
        # M matrix
        self.ax_M = self.fig.add_subplot(gs[0, 2])
        self.ax_M.set_facecolor('#16213e')
        
        # C matrix
        self.ax_C = self.fig.add_subplot(gs[0, 3])
        self.ax_C.set_facecolor('#16213e')
        
        # g vector and torque breakdown
        self.ax_breakdown = self.fig.add_subplot(gs[1, 2:4])
        self.ax_breakdown.set_facecolor('#16213e')
        
        # Torque bar chart
        self.ax_torques = self.fig.add_subplot(gs[2, :])
        self.ax_torques.set_facecolor('#16213e')
        
        self.setup_controls()
        self.update_plot()
        
    def setup_controls(self):
        """Setup sliders for q, q̇, q̈."""
        slider_width = 0.08
        slider_height = 0.012
        textbox_width = 0.04
        textbox_height = 0.022
        gap = 0.006
        
        y_base = 0.32
        y_step = 0.038
        
        # Column positions
        col1_x = 0.03   # q (position)
        col2_x = 0.26   # q̇ (velocity)
        col3_x = 0.49   # q̈ (acceleration)
        col4_x = 0.72   # Presets
        
        slider_start = 0.055
        text_offset = slider_width + gap
        
        # Headers
        self.fig.text(col1_x + 0.08, y_base + 0.055, 'POSITION q', 
                     ha='center', fontsize=12, fontweight='bold', color='#e94560')
        self.fig.text(col2_x + 0.08, y_base + 0.055, 'VELOCITY q̇', 
                     ha='center', fontsize=12, fontweight='bold', color='#00ff88')
        self.fig.text(col3_x + 0.08, y_base + 0.055, 'ACCELERATION q̈', 
                     ha='center', fontsize=12, fontweight='bold', color='#ffd700')
        self.fig.text(col4_x + 0.08, y_base + 0.055, 'PRESETS', 
                     ha='center', fontsize=12, fontweight='bold', color='#00bcd4')
        
        # Joint labels
        joint_labels = ['θ₁', 'd₂', 'θ₃', 'θ₄']
        units_q = ['(°)', '(m)', '(°)', '(°)']
        units_qd = ['(°/s)', '(m/s)', '(°/s)', '(°/s)']
        units_qdd = ['(°/s²)', '(m/s²)', '(°/s²)', '(°/s²)']
        
        # Slider ranges
        q_ranges = [(-180, 180), (0, 0.5), (-90, 90), (-150, 150)]
        qd_ranges = [(-60, 60), (-0.2, 0.2), (-60, 60), (-60, 60)]
        qdd_ranges = [(-100, 100), (-0.5, 0.5), (-100, 100), (-100, 100)]
        
        self.sliders_q = []
        self.sliders_qd = []
        self.sliders_qdd = []
        self.texts_q = []
        self.texts_qd = []
        self.texts_qdd = []
        
        for i in range(4):
            y = y_base - i * y_step
            
            # Position sliders (q)
            self.fig.text(col1_x, y, f'{joint_labels[i]} {units_q[i]}', 
                         color='white', fontsize=9, va='center', fontweight='bold')
            ax_q = self.fig.add_axes([col1_x + slider_start, y - slider_height/2, slider_width, slider_height])
            ax_q.set_facecolor('#0f3460')
            
            if i == 1:  # d2 is in meters
                init_val = self.q[i]
            else:
                init_val = np.degrees(self.q[i])
            
            slider_q = Slider(ax_q, '', q_ranges[i][0], q_ranges[i][1], valinit=init_val, color='#e94560')
            slider_q.valtext.set_visible(False)
            self.sliders_q.append(slider_q)
            
            ax_q_text = self.fig.add_axes([col1_x + slider_start + text_offset, y - textbox_height/2, 
                                           textbox_width, textbox_height])
            ax_q_text.set_facecolor('#e8e8e8')
            text_q = TextBox(ax_q_text, '', initial=f'{init_val:.1f}', textalignment='center')
            text_q.text_disp.set_color('#1a1a2e')
            text_q.text_disp.set_fontsize(10)
            text_q.text_disp.set_fontweight('bold')
            self.texts_q.append(text_q)
            
            # Velocity sliders (q̇)
            self.fig.text(col2_x, y, f'{joint_labels[i]}̇ {units_qd[i]}', 
                         color='white', fontsize=9, va='center', fontweight='bold')
            ax_qd = self.fig.add_axes([col2_x + slider_start, y - slider_height/2, slider_width, slider_height])
            ax_qd.set_facecolor('#0f3460')
            
            if i == 1:
                init_val_d = self.qd[i]
            else:
                init_val_d = np.degrees(self.qd[i])
            
            slider_qd = Slider(ax_qd, '', qd_ranges[i][0], qd_ranges[i][1], valinit=init_val_d, color='#00ff88')
            slider_qd.valtext.set_visible(False)
            self.sliders_qd.append(slider_qd)
            
            ax_qd_text = self.fig.add_axes([col2_x + slider_start + text_offset, y - textbox_height/2, 
                                            textbox_width, textbox_height])
            ax_qd_text.set_facecolor('#e8e8e8')
            text_qd = TextBox(ax_qd_text, '', initial=f'{init_val_d:.1f}', textalignment='center')
            text_qd.text_disp.set_color('#1a1a2e')
            text_qd.text_disp.set_fontsize(10)
            text_qd.text_disp.set_fontweight('bold')
            self.texts_qd.append(text_qd)
            
            # Acceleration sliders (q̈)
            self.fig.text(col3_x, y, f'{joint_labels[i]}̈ {units_qdd[i]}', 
                         color='white', fontsize=9, va='center', fontweight='bold')
            ax_qdd = self.fig.add_axes([col3_x + slider_start, y - slider_height/2, slider_width, slider_height])
            ax_qdd.set_facecolor('#0f3460')
            
            if i == 1:
                init_val_dd = self.qdd[i]
            else:
                init_val_dd = np.degrees(self.qdd[i])
            
            slider_qdd = Slider(ax_qdd, '', qdd_ranges[i][0], qdd_ranges[i][1], valinit=init_val_dd, color='#ffd700')
            slider_qdd.valtext.set_visible(False)
            self.sliders_qdd.append(slider_qdd)
            
            ax_qdd_text = self.fig.add_axes([col3_x + slider_start + text_offset, y - textbox_height/2, 
                                             textbox_width, textbox_height])
            ax_qdd_text.set_facecolor('#e8e8e8')
            text_qdd = TextBox(ax_qdd_text, '', initial=f'{init_val_dd:.1f}', textalignment='center')
            text_qdd.text_disp.set_color('#1a1a2e')
            text_qdd.text_disp.set_fontsize(10)
            text_qdd.text_disp.set_fontweight('bold')
            self.texts_qdd.append(text_qdd)
        
        # Preset buttons
        btn_width = 0.12
        btn_height = 0.035
        btn_x = col4_x + 0.02
        
        ax_static = self.fig.add_axes([btn_x, y_base, btn_width, btn_height])
        self.btn_static = Button(ax_static, 'Static', color='#0f3460', hovercolor='#e94560')
        self.btn_static.label.set_color('white')
        self.btn_static.label.set_fontsize(10)
        
        ax_const_vel = self.fig.add_axes([btn_x, y_base - y_step, btn_width, btn_height])
        self.btn_const_vel = Button(ax_const_vel, 'Const Velocity', color='#0f3460', hovercolor='#00ff88')
        self.btn_const_vel.label.set_color('white')
        self.btn_const_vel.label.set_fontsize(10)
        
        ax_accel = self.fig.add_axes([btn_x, y_base - 2*y_step, btn_width, btn_height])
        self.btn_accel = Button(ax_accel, 'Accelerating', color='#0f3460', hovercolor='#ffd700')
        self.btn_accel.label.set_color('white')
        self.btn_accel.label.set_fontsize(10)
        
        ax_home = self.fig.add_axes([btn_x, y_base - 3*y_step, btn_width, btn_height])
        self.btn_home = Button(ax_home, 'Home', color='#0f3460', hovercolor='#00bcd4')
        self.btn_home.label.set_color('white')
        self.btn_home.label.set_fontsize(10)
        
        # Connect callbacks
        for i in range(4):
            self.sliders_q[i].on_changed(lambda val, idx=i: self.on_slider_q(idx, val))
            self.sliders_qd[i].on_changed(lambda val, idx=i: self.on_slider_qd(idx, val))
            self.sliders_qdd[i].on_changed(lambda val, idx=i: self.on_slider_qdd(idx, val))
            self.texts_q[i].on_submit(lambda val, idx=i: self.on_text_q(idx, val))
            self.texts_qd[i].on_submit(lambda val, idx=i: self.on_text_qd(idx, val))
            self.texts_qdd[i].on_submit(lambda val, idx=i: self.on_text_qdd(idx, val))
        
        self.btn_static.on_clicked(self.preset_static)
        self.btn_const_vel.on_clicked(self.preset_const_vel)
        self.btn_accel.on_clicked(self.preset_accel)
        self.btn_home.on_clicked(self.preset_home)
        
        # ====== Mass Parameters Section ======
        mass_y = 0.03
        mass_slider_width = 0.10
        
        # Separator line
        ax_sep = self.fig.add_axes([0.02, 0.075, 0.96, 0.002])
        ax_sep.set_facecolor('#ff6b6b')
        ax_sep.set_xticks([])
        ax_sep.set_yticks([])
        for spine in ax_sep.spines.values():
            spine.set_visible(False)
        
        self.fig.text(0.02, mass_y + 0.035, 'ROBOT PARAMETERS', 
                     ha='left', fontsize=11, fontweight='bold', color='#ff6b6b')
        
        # m1 - Slider mass
        self.fig.text(0.02, mass_y, 'm₁ (slider):', color='white', fontsize=9, va='center')
        ax_m1 = self.fig.add_axes([0.10, mass_y - 0.008, mass_slider_width, 0.015])
        ax_m1.set_facecolor('#0f3460')
        self.slider_m1 = Slider(ax_m1, '', 0.5, 5.0, valinit=self.m1, color='#ff6b6b')
        self.slider_m1.valtext.set_visible(False)
        self.fig.text(0.21, mass_y, f'{self.m1:.1f} kg', color='#ff6b6b', fontsize=10, 
                     fontweight='bold', va='center')
        self.text_m1 = self.fig.texts[-1]
        
        # m2 - Upper arm mass
        self.fig.text(0.24, mass_y, 'm₂ (arm):', color='white', fontsize=9, va='center')
        ax_m2 = self.fig.add_axes([0.31, mass_y - 0.008, mass_slider_width, 0.015])
        ax_m2.set_facecolor('#0f3460')
        self.slider_m2 = Slider(ax_m2, '', 0.5, 5.0, valinit=self.m2, color='#ff6b6b')
        self.slider_m2.valtext.set_visible(False)
        self.fig.text(0.42, mass_y, f'{self.m2:.1f} kg', color='#ff6b6b', fontsize=10, 
                     fontweight='bold', va='center')
        self.text_m2 = self.fig.texts[-1]
        
        # m3 - Forearm mass
        self.fig.text(0.46, mass_y, 'm₃ (forearm):', color='white', fontsize=9, va='center')
        ax_m3 = self.fig.add_axes([0.54, mass_y - 0.008, mass_slider_width, 0.015])
        ax_m3.set_facecolor('#0f3460')
        self.slider_m3 = Slider(ax_m3, '', 0.2, 3.0, valinit=self.m3, color='#ff6b6b')
        self.slider_m3.valtext.set_visible(False)
        self.fig.text(0.65, mass_y, f'{self.m3:.1f} kg', color='#ff6b6b', fontsize=10, 
                     fontweight='bold', va='center')
        self.text_m3 = self.fig.texts[-1]
        
        # Gravity
        self.fig.text(0.69, mass_y, 'gravity:', color='white', fontsize=9, va='center')
        ax_g = self.fig.add_axes([0.75, mass_y - 0.008, 0.08, 0.015])
        ax_g.set_facecolor('#0f3460')
        self.slider_g = Slider(ax_g, '', 0, 20.0, valinit=self.g, color='#ffd700')
        self.slider_g.valtext.set_visible(False)
        self.fig.text(0.84, mass_y, f'{self.g:.1f} m/s²', color='#ffd700', fontsize=10, 
                     fontweight='bold', va='center')
        self.text_g = self.fig.texts[-1]
        
        # Total mass display
        total_mass = self.m1 + self.m2 + self.m3
        self.fig.text(0.91, mass_y, f'Σm = {total_mass:.1f} kg', color='cyan', fontsize=11, 
                     fontweight='bold', va='center')
        self.text_total_mass = self.fig.texts[-1]
        
        # Connect mass slider callbacks
        self.slider_m1.on_changed(self.on_mass_change)
        self.slider_m2.on_changed(self.on_mass_change)
        self.slider_m3.on_changed(self.on_mass_change)
        self.slider_g.on_changed(self.on_gravity_change)
    
    def update_text_display(self, text_widget, value):
        """Update text display without triggering on_submit callback."""
        # Directly update the text display
        text_widget.text_disp.set_text(value)
    
    def on_slider_q(self, idx, val):
        if self._updating:
            return
        self._updating = True
        try:
            if idx == 1:
                self.q[idx] = val
                self.update_text_display(self.texts_q[idx], f'{val:.2f}')
            else:
                self.q[idx] = np.radians(val)
                self.update_text_display(self.texts_q[idx], f'{val:.1f}')
            self.update_plot()
        finally:
            self._updating = False
    
    def on_slider_qd(self, idx, val):
        if self._updating:
            return
        self._updating = True
        try:
            if idx == 1:
                self.qd[idx] = val
                self.update_text_display(self.texts_qd[idx], f'{val:.3f}')
            else:
                self.qd[idx] = np.radians(val)
                self.update_text_display(self.texts_qd[idx], f'{val:.1f}')
            self.update_plot()
        finally:
            self._updating = False
    
    def on_slider_qdd(self, idx, val):
        if self._updating:
            return
        self._updating = True
        try:
            if idx == 1:
                self.qdd[idx] = val
                self.update_text_display(self.texts_qdd[idx], f'{val:.2f}')
            else:
                self.qdd[idx] = np.radians(val)
                self.update_text_display(self.texts_qdd[idx], f'{val:.1f}')
            self.update_plot()
        finally:
            self._updating = False
    
    def on_text_q(self, idx, val):
        if self._updating:
            return
        self._updating = True
        try:
            v = float(val)
            if idx == 1:
                v = np.clip(v, 0, 0.5)
                self.q[idx] = v
            else:
                v = np.clip(v, -180 if idx == 0 else -90 if idx == 2 else -150, 
                           180 if idx == 0 else 90 if idx == 2 else 150)
                self.q[idx] = np.radians(v)
            self.sliders_q[idx].set_val(v)
            self.update_plot()
        except ValueError:
            pass
        finally:
            self._updating = False
    
    def on_text_qd(self, idx, val):
        if self._updating:
            return
        self._updating = True
        try:
            v = float(val)
            if idx == 1:
                v = np.clip(v, -0.2, 0.2)
                self.qd[idx] = v
            else:
                v = np.clip(v, -60, 60)
                self.qd[idx] = np.radians(v)
            self.sliders_qd[idx].set_val(v)
            self.update_plot()
        except ValueError:
            pass
        finally:
            self._updating = False
    
    def on_text_qdd(self, idx, val):
        if self._updating:
            return
        self._updating = True
        try:
            v = float(val)
            if idx == 1:
                v = np.clip(v, -0.5, 0.5)
                self.qdd[idx] = v
            else:
                v = np.clip(v, -100, 100)
                self.qdd[idx] = np.radians(v)
            self.sliders_qdd[idx].set_val(v)
            self.update_plot()
        except ValueError:
            pass
        finally:
            self._updating = False
    
    def set_all_sliders(self, q, qd, qdd):
        """Set all slider values."""
        self._updating = True
        try:
            for i in range(4):
                if i == 1:
                    self.sliders_q[i].set_val(q[i])
                    self.sliders_qd[i].set_val(qd[i])
                    self.sliders_qdd[i].set_val(qdd[i])
                    self.update_text_display(self.texts_q[i], f'{q[i]:.2f}')
                    self.update_text_display(self.texts_qd[i], f'{qd[i]:.3f}')
                    self.update_text_display(self.texts_qdd[i], f'{qdd[i]:.2f}')
                else:
                    self.sliders_q[i].set_val(np.degrees(q[i]))
                    self.sliders_qd[i].set_val(np.degrees(qd[i]))
                    self.sliders_qdd[i].set_val(np.degrees(qdd[i]))
                    self.update_text_display(self.texts_q[i], f'{np.degrees(q[i]):.1f}')
                    self.update_text_display(self.texts_qd[i], f'{np.degrees(qd[i]):.1f}')
                    self.update_text_display(self.texts_qdd[i], f'{np.degrees(qdd[i]):.1f}')
            self.update_plot()
        finally:
            self._updating = False
    
    def preset_static(self, event):
        """Static equilibrium - only gravity."""
        q = np.array([np.radians(30), 0.15, np.radians(45), np.radians(30)])
        qd = np.array([0.0, 0.0, 0.0, 0.0])
        qdd = np.array([0.0, 0.0, 0.0, 0.0])
        self.q, self.qd, self.qdd = q.copy(), qd.copy(), qdd.copy()
        self.set_all_sliders(q, qd, qdd)
    
    def preset_const_vel(self, event):
        """Constant velocity - gravity + Coriolis."""
        q = np.array([np.radians(30), 0.15, np.radians(45), np.radians(30)])
        qd = np.array([np.radians(20), 0.05, np.radians(30), np.radians(20)])
        qdd = np.array([0.0, 0.0, 0.0, 0.0])
        self.q, self.qd, self.qdd = q.copy(), qd.copy(), qdd.copy()
        self.set_all_sliders(q, qd, qdd)
    
    def preset_accel(self, event):
        """Accelerating - full dynamics."""
        q = np.array([np.radians(0), 0.10, np.radians(30), np.radians(0)])
        qd = np.array([np.radians(10), 0.02, np.radians(15), np.radians(10)])
        qdd = np.array([np.radians(50), 0.1, np.radians(60), np.radians(40)])
        self.q, self.qd, self.qdd = q.copy(), qd.copy(), qdd.copy()
        self.set_all_sliders(q, qd, qdd)
    
    def preset_home(self, event):
        """Home position, at rest."""
        q = np.array([0.0, 0.1, 0.0, 0.0])
        qd = np.array([0.0, 0.0, 0.0, 0.0])
        qdd = np.array([0.0, 0.0, 0.0, 0.0])
        self.q, self.qd, self.qdd = q.copy(), qd.copy(), qdd.copy()
        self.set_all_sliders(q, qd, qdd)
    
    def on_mass_change(self, val):
        """Handle mass parameter changes."""
        if self._updating:
            return
        self._updating = True
        try:
            # Update mass values
            self.m1 = self.slider_m1.val
            self.m2 = self.slider_m2.val
            self.m3 = self.slider_m3.val
            
            # Update inertias
            self.I2 = (1/12) * self.m2 * self.L1**2
            self.I3 = (1/12) * self.m3 * self.L2**2
            
            # Update text displays
            self.text_m1.set_text(f'{self.m1:.1f} kg')
            self.text_m2.set_text(f'{self.m2:.1f} kg')
            self.text_m3.set_text(f'{self.m3:.1f} kg')
            self.text_total_mass.set_text(f'Σ = {self.m1 + self.m2 + self.m3:.1f} kg')
            
            # Rebuild robot with new masses
            self.build_robot()
            self.update_plot()
        finally:
            self._updating = False
    
    def on_gravity_change(self, val):
        """Handle gravity parameter change."""
        if self._updating:
            return
        self._updating = True
        try:
            self.g = self.slider_g.val
            self.text_g.set_text(f'{self.g:.1f} m/s²')
            
            # Rebuild robot with new gravity
            self.build_robot()
            self.update_plot()
        finally:
            self._updating = False
    
    def update_plot(self):
        """Update all plots."""
        # Compute dynamics
        M = self.robot.inertia(self.q)
        C = self.robot.coriolis(self.q, self.qd)
        g = self.robot.gravload(self.q)
        
        M_qdd = M @ self.qdd
        C_qd = C @ self.qd
        tau = M_qdd + C_qd + g
        
        # Get joint positions
        p0, p1, p2, p3, p4, p5 = self.get_joint_positions(self.q)
        
        # ====== 3D Robot Plot ======
        self.ax_robot.clear()
        self.ax_robot.set_facecolor('#16213e')
        
        # Base
        theta_base = np.linspace(0, 2*np.pi, 30)
        r_base = 0.12
        x_base = r_base * np.cos(theta_base)
        y_base = r_base * np.sin(theta_base)
        self.ax_robot.plot(x_base, y_base, np.zeros_like(theta_base), color='gray', linewidth=2)
        
        # Links
        self.ax_robot.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 
                          color='#4a90d9', linewidth=10, solid_capstyle='round')
        self.ax_robot.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                          color='#00bcd4', linewidth=12, solid_capstyle='round')
        self.ax_robot.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 
                          color='#e94560', linewidth=8, solid_capstyle='round')
        self.ax_robot.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]], 
                          color='#ff9800', linewidth=6, solid_capstyle='round')
        
        # Joints
        for pos, color in zip([p1, p2, p3, p4], ['#4a90d9', '#00bcd4', '#e94560', '#ff9800']):
            self.ax_robot.scatter(*pos, color=color, s=100, edgecolors='white', linewidths=2, zorder=5)
        self.ax_robot.scatter(*p5, color='#00ff88', s=150, marker='*', edgecolors='white', linewidths=1.5, zorder=6)
        
        # Gravity arrow
        self.ax_robot.quiver(p5[0], p5[1], p5[2], 0, 0, -0.15, color='yellow', 
                            arrow_length_ratio=0.2, linewidth=2, label='g')
        
        max_reach = self.L0 + 0.5 + self.L1 + self.L2 + 0.2
        self.ax_robot.set_xlim([-max_reach, max_reach])
        self.ax_robot.set_ylim([-max_reach, max_reach])
        self.ax_robot.set_zlim([0, max_reach])
        self.ax_robot.set_xlabel('X', color='white', fontsize=10)
        self.ax_robot.set_ylabel('Y', color='white', fontsize=10)
        self.ax_robot.set_zlabel('Z', color='white', fontsize=10)
        self.ax_robot.set_title(f'EE: ({p5[0]:.3f}, {p5[1]:.3f}, {p5[2]:.3f}) m', 
                               color='white', fontsize=11)
        self.ax_robot.tick_params(colors='white', labelsize=8)
        self.ax_robot.xaxis.pane.fill = False
        self.ax_robot.yaxis.pane.fill = False
        self.ax_robot.zaxis.pane.fill = False
        
        # ====== Mass Matrix M(q) ======
        self.ax_M.clear()
        self.ax_M.set_facecolor('#16213e')
        self.ax_M.axis('off')
        self.ax_M.set_title('M(q) - Mass Matrix', color='white', fontsize=11, fontweight='bold')
        
        cell_text_M = [[f'{M[i,j]:.3f}' for j in range(4)] for i in range(4)]
        table_M = self.ax_M.table(cellText=cell_text_M, loc='center', cellLoc='center')
        table_M.auto_set_font_size(False)
        table_M.set_fontsize(9)
        table_M.scale(1.0, 1.8)
        
        max_M = np.max(np.abs(M)) + 0.01
        for i in range(4):
            for j in range(4):
                cell = table_M[(i, j)]
                val = M[i,j] / max_M
                cell.set_facecolor(plt.cm.Reds(0.2 + 0.6 * abs(val)))
                cell.set_text_props(color='black', fontweight='bold')
        
        # ====== Coriolis Matrix C(q,q̇) ======
        self.ax_C.clear()
        self.ax_C.set_facecolor('#16213e')
        self.ax_C.axis('off')
        self.ax_C.set_title('C(q,q̇) - Coriolis', color='white', fontsize=11, fontweight='bold')
        
        cell_text_C = [[f'{C[i,j]:.3f}' for j in range(4)] for i in range(4)]
        table_C = self.ax_C.table(cellText=cell_text_C, loc='center', cellLoc='center')
        table_C.auto_set_font_size(False)
        table_C.set_fontsize(9)
        table_C.scale(1.0, 1.8)
        
        max_C = np.max(np.abs(C)) + 0.001
        for i in range(4):
            for j in range(4):
                cell = table_C[(i, j)]
                val = C[i,j] / max_C if max_C > 0.001 else 0
                cell.set_facecolor(plt.cm.Greens(0.2 + 0.5 * abs(val)))
                cell.set_text_props(color='black', fontweight='bold')
        
        # ====== Breakdown Table ======
        self.ax_breakdown.clear()
        self.ax_breakdown.set_facecolor('#16213e')
        self.ax_breakdown.axis('off')
        self.ax_breakdown.set_title('τ = M(q)q̈ + C(q,q̇)q̇ + g(q)', color='white', fontsize=12, fontweight='bold')
        
        labels = ['τ₁ (Nm)', 'F₂ (N)', 'τ₃ (Nm)', 'τ₄ (Nm)']
        col_labels = ['Joint', 'M·q̈', 'C·q̇', 'g(q)', '= τ']
        
        cell_text = []
        for i in range(4):
            cell_text.append([labels[i], f'{M_qdd[i]:.3f}', f'{C_qd[i]:.3f}', 
                            f'{g[i]:.3f}', f'{tau[i]:.3f}'])
        
        table_bd = self.ax_breakdown.table(cellText=cell_text, colLabels=col_labels,
                                           loc='center', cellLoc='center')
        table_bd.auto_set_font_size(False)
        table_bd.set_fontsize(10)
        table_bd.scale(1.0, 1.8)
        
        # Color the table
        for j in range(5):
            table_bd[(0, j)].set_facecolor('#0f3460')
            table_bd[(0, j)].set_text_props(color='white', fontweight='bold')
        
        colors = ['#2d3a4a', '#e94560', '#00ff88', '#ffd700', '#00bcd4']
        for i in range(4):
            for j in range(5):
                table_bd[(i+1, j)].set_facecolor(colors[j] if j > 0 else '#2d3a4a')
                table_bd[(i+1, j)].set_text_props(color='white' if j == 0 else 'black', fontweight='bold')
        
        # ====== Torque Bar Chart ======
        self.ax_torques.clear()
        self.ax_torques.set_facecolor('#16213e')
        
        x = np.arange(4)
        width = 0.2
        
        bars1 = self.ax_torques.bar(x - 1.5*width, M_qdd, width, label='M·q̈ (Inertia)', color='#e94560', edgecolor='white')
        bars2 = self.ax_torques.bar(x - 0.5*width, C_qd, width, label='C·q̇ (Coriolis)', color='#00ff88', edgecolor='white')
        bars3 = self.ax_torques.bar(x + 0.5*width, g, width, label='g(q) (Gravity)', color='#ffd700', edgecolor='white')
        bars4 = self.ax_torques.bar(x + 1.5*width, tau, width, label='τ (Total)', color='#00bcd4', edgecolor='white', linewidth=2)
        
        self.ax_torques.set_xticks(x)
        self.ax_torques.set_xticklabels(['τ₁ (Nm)', 'F₂ (N)', 'τ₃ (Nm)', 'τ₄ (Nm)'], color='white', fontsize=11)
        self.ax_torques.set_ylabel('Torque / Force', color='white', fontsize=11)
        self.ax_torques.tick_params(colors='white', labelsize=10)
        self.ax_torques.legend(loc='upper right', fontsize=9, facecolor='#16213e', edgecolor='white', labelcolor='white')
        self.ax_torques.axhline(y=0, color='white', linewidth=0.5)
        self.ax_torques.grid(True, alpha=0.3, axis='y', color='white')
        
        for spine in self.ax_torques.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        plt.show()


def main():
    print("=" * 65)
    print("   RPRR Robot Dynamics Simulator")
    print("   τ = M(q)q̈ + C(q,q̇)q̇ + g(q)")
    print("=" * 65)
    print("\n  Controls:")
    print("  • q  (Position)     - Joint angles/displacement")
    print("  • q̇  (Velocity)     - Joint velocities")
    print("  • q̈  (Acceleration) - Joint accelerations")
    print("\n  Presets:")
    print("  • Static       - q̇=0, q̈=0 (gravity only)")
    print("  • Const Vel    - q̈=0 (gravity + Coriolis)")
    print("  • Accelerating - Full dynamics")
    print("=" * 65)
    
    sim = RPRRDynamicsSimulator()
    sim.run()


if __name__ == "__main__":
    main()
