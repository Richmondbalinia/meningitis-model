# Meningitis Transmission Dynamics and Stability Analysis

This repository contains Python scripts for simulating the transmission dynamics of meningitis, with a focus on intervention strategies and the stability of disease equilibria. The models incorporate vaccination (including booster doses), antibiotic resistance, and key epidemiological parameters.

## Contents

- **intervention_simulation.py**  
  Implements a deterministic compartmental model using the Runge-Kutta 4th order method (RK4) to assess the impact of varying:
  - Vaccination rates
  - Booster uptake
  - Resistance mutation
  - Recovery rates  
  Outputs include JPEG plots showing how each parameter affects susceptible and infected populations.

- **stability_analysis.py**  
  Evaluates the stability of the **Disease-Free Equilibrium (DFE)** and **Endemic Equilibrium (EE)** under different transmission intensities.  
  Results are visualized to show infection trends under small perturbations near equilibrium.

## Getting Started

### Requirements
- Python 3.7+
- numpy
- matplotlib

To install the required packages:

```bash
pip install numpy matplotlib
python intervention_simulation.py
python stability_analysis.py

