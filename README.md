Code accompanying the paper:

**Optimal sensor placement for the reconstruction of ocean states using differentiable Gumbel-Softmax sampling operator**

## Overview

This repository provides the implementation of a differentiable framework for optimal sensor placement in ocean-state reconstruction under strict observation-budget constraints.

The method combines:

- a **Gumbel-Softmax-based differentiable sampling operator**
- a **budget-aware probabilistic observation mask**
- a **joint optimization** of sensor placement and reconstruction parameters
- an application to **Sea Surface Height (SSH)** reconstruction using **Optimal Interpolation (OI)**

The framework is designed for **adaptive observation network design** in geophysical systems and is evaluated through **Observing System Simulation Experiments (OSSEs)** on Gulf Stream SSH data.
