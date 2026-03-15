# Inverse Kinematics Visualizer

## Overview
A browser-based interactive IK chain demo built with Three.js or p5.js. Users drag an end-effector and watch the joint chain solve in real-time using CCD (Cyclic Coordinate Descent) or FABRIK.

## Category
Robotics / Interactive Visualization / Web

## Stack
- **p5.js** (simpler, 2D) or **Three.js** (3D, more impressive)
- Vanilla JavaScript — no framework needed
- Hosted on GitHub Pages (zero cost, instant deployment)

## Key Features
- Drag end-effector with mouse/touch
- Joint chain follows in real-time
- Configurable: number of joints, segment lengths, joint angle limits
- Toggle between CCD and FABRIK solvers
- Show/hide: joint angles, reach circle, constraint cones
- Mobile-friendly (touch drag support)

## IK Algorithms
### FABRIK (Forward And Backward Reaching Inverse Kinematics)
- Fast, stable, handles constraints well
- Iterative: reach forward to target, then backward to root

### CCD (Cyclic Coordinate Descent)
- Simpler to implement
- Rotates each joint one at a time to minimize end-effector error

## Architecture
- Chain represented as array of joint objects `{angle, length}`
- Each animation frame: run IK solver → update joint transforms → render
- p5.js draw loop or requestAnimationFrame handles real-time rendering

## Portfolio Value
- Extremely eye-catching in a portfolio
- Directly relevant to prosthetics / robotics arm control
- Shows math + programming + UI skills in one demo

## Milestones
- [ ] Basic 2D chain rendering with p5.js
- [ ] Mouse drag interaction
- [ ] FABRIK solver
- [ ] CCD solver + toggle
- [ ] Joint angle constraints
- [ ] Deploy on GitHub Pages
- [ ] README + embed in portfolio site

## Notes
<!-- Add implementation notes, decisions, and progress here -->
