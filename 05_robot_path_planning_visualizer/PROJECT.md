# 2D Robot Path Planning Visualizer

## Overview
An interactive demo that visualizes classical path planning algorithms (A*, RRT, PRM) on a 2D grid with obstacles. Users can draw walls and watch the algorithms explore the space in real-time.

## Category
Robotics / Algorithms / Interactive Visualization

## Stack
- **Pygame** (desktop) or **React + HTML Canvas** (browser)
- Python (Pygame version) or JavaScript (browser version)
- No external ML dependencies required

## Algorithms
1. **A*** — optimal grid-based search with heuristic
2. **RRT** (Rapidly-exploring Random Tree) — sampling-based, good for continuous spaces
3. **PRM** (Probabilistic Roadmap Method) — roadmap-based, good for multi-query planning

## Key Features
- Click to place/remove obstacles on the grid
- Set start and goal positions
- Select algorithm from dropdown
- Animated step-by-step exploration
- Display: path length, nodes explored, computation time
- Speed control slider

## Architecture
- Grid representation (NumPy array or JS 2D array)
- Each algorithm implemented as a generator (yields intermediate states for animation)
- Renderer draws grid, obstacles, explored nodes, and final path

## Portfolio Value
- Visually demonstrates algorithms coursework knowledge
- Immediately impressive in a live demo
- Relevant to robotics/navigation roles

## Milestones
- [ ] Grid + obstacle drawing UI
- [ ] A* implementation with animation
- [ ] RRT implementation with animation
- [ ] PRM implementation with animation
- [ ] Algorithm comparison stats panel
- [ ] README + demo GIF

## Notes
<!-- Add implementation notes, decisions, and progress here -->
