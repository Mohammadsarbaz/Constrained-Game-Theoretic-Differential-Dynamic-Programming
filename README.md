# Constrained Game-Theoretic Differential Dynamic Programming (CGT-DDP)

Python and MATLAB implementation of **Constrained Game-Theoretic Differential Dynamic Programming (CGT-DDP)** for continuous-time nonlinear systems subject to both point-wise and functional constraints.

---

## 📖 Overview

CGT-DDP extends the standard Game-Theoretic DDP (GT-DDP) framework to handle two classes of constraints that arise in practical control problems:

- **Point-wise constraints** — instantaneous restrictions on states and control inputs at each time step (e.g., actuator bounds, velocity limits)
- **Functional constraints** — trajectory-dependent requirements accumulated over the entire horizon (e.g., total path length, obstacle-avoidance budget, energy consumption)

The framework formulates the problem as a finite-horizon two-player min-max differential game, where one player minimizes a cost function while the other maximizes it. Functional constraints are handled through an augmented Lagrangian formulation, while point-wise constraints are enforced via an active-set quadratic programming method at each time step.

---

## 🔑 Key Contributions

1. A **CGT-DDP framework** for continuous-time nonlinear differential games with both point-wise and functional constraints
2. **Point-wise constraints** incorporated into the min-max optimization via active-set QP at each backward pass step
3. **Functional constraints** handled through augmented state variables and Lagrangian multipliers
4. Derivation of optimal minimizing and maximizing **control update laws** and **backward propagation equations** for the value function
5. **Convergence and stability proofs** for the proposed algorithm
6. Demonstrated **linear computational complexity O(N)** vs. cubic complexity O(N³) of centralized NLP solvers

---

## 📁 Repository Structure

### 1. `2D_Quadrotor_Obstacle_Avoidance` — 2D Quadrotor with Functional Constraint
CGT-DDP applied to a planar quadrotor system navigating around a circular obstacle. The obstacle-avoidance requirement is enforced as a **functional constraint** through an augmented state. Point-wise bounds are imposed on both minimizing and maximizing control inputs (−4 ≤ u ≤ 4, −1 ≤ v ≤ 1).

### 2. `3D_Quadrotor_Navigation` — 3D Quadrotor with Functional + Point-Wise Constraints
CGT-DDP applied to a full 3D quadrotor navigation task. A **functional constraint** limits the accumulated navigation-path measure (z(T) ≤ 20), while **point-wise constraints** bound the minimizing and maximizing control inputs throughout the horizon.

### 3. `Pursuit_Evasion_Game` — Pursuit-Evasion with Obstacle Avoidance
CGT-DDP applied to a two-player pursuit-evasion differential game. The pursuer minimizes distance to the evader while avoiding a forbidden obstacle region enforced through a **functional constraint** via an augmented state.

---

## 🛠️ Requirements
- Python 3.8 or later
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## 🚀 How to Run
1. Clone or download this repository
2. Navigate to the desired example folder
3. Run the main Python script:
```bash
python main.py
```

## 📊 Results
Figures and simulation results are available inside each example folder.

## 🔬 Algorithm Summary

The CGT-DDP algorithm proceeds as follows:
1. **Initialize** nominal control trajectories for both players
2. **Backward pass** — integrate backward ODEs for value function and its derivatives
3. **Point-wise constraint enforcement** — solve local QP over active constraint set at each time step
4. **Forward pass** — simulate nonlinear dynamics forward with updated controls
5. **Functional constraint update** — update Lagrange multipliers based on constraint violation
6. **Repeat** until convergence

---

## 📄 Related Paper
> **Mohammad Sarbaz**, Wei Sun. *Constrained Game Theoretic Trajectory Optimization in Continuous Time.* Journal of Optimization Theory and Applications, 2026.

## 📬 Contact
**Mohammad Sarbaz** — mohammad.sarbaz@ou.edu
🔗 [LinkedIn](https://www.linkedin.com/in/mohammad-sarbaz-94256b1b7/) | 🎓 [Google Scholar](https://scholar.google.com/citations?user=St87OnMAAAAJ&hl=en&oi=ao)
