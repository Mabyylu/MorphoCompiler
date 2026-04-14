# MorphoCompiler: Differentiable Robot Co-Design System

[![Research Grade](https://img.shields.io/badge/status-research--grade-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**A frontier research project combining computer science, advanced programming, AI, robotics, and scientific innovation.**

## Overview

MorphoCompiler is a **differentiable physics engine** that enables end-to-end co-optimization of robot morphology (body) and control policy (brain). This is a fundamentally new approach to robot design where physical parameters are treated as differentiable tensors alongside neural network weights.

### Core Innovation

Traditional robot design follows a sequential process:
1. Humans design the robot body
2. Engineers program the controller
3. Iterative tuning (if time permits)

MorphoCompiler enables **simultaneous gradient-based optimization** of both body and brain, discovering non-intuitive morphologies that humans would never conceive.

```
Loss = f(morphology_params, neural_weights)
∂Loss/∂morphology ← Gradient descent grows optimal body
∂Loss/∂control ← Gradient descent learns optimal controller
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MorphoCompiler Stack                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Fabrication Pipeline                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ OpenSCAD │  │   URDF   │  │   BOM    │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Optimization Engine                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Adam Optimizer + JAX Autodiff                        │   │
│  │  ∇ morphology ←→ ∇ control                            │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Differentiable Physics                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Forward  │  │ Contact  │  │ Dynamics │                  │
│  │ Kinematics│ │ Model    │  │ Engine   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: MorphoDSL                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MorphoChain().add_link().add_joint().build()        │   │
│  │  All parameters are JAX tensors                       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- JAX/JAXlib (GPU recommended for serious optimization)
- Optional: Flax, Optax for advanced features

### Quick Start

```bash
# Clone repository
cd /workspace/morpho_compiler

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run main optimization demo
python src/main.py
```

## Project Structure

```
morpho_compiler/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── morphodsl.py          # Domain-Specific Language for morphology
│   ├── physics_engine.py     # Differentiable contact dynamics
│   ├── control_policy.py     # Neural network controllers
│   ├── fabrication.py        # Morphology-to-CAD pipeline
│   └── main.py               # Main optimization loop
├── tests/
│   ├── test_morphodsl.py     # DSL unit tests
│   ├── test_physics.py       # Physics engine tests
│   └── test_gradients.py     # Gradient validation tests
├── benchmarks/
│   └── locomotion_benchmark.py  # Standard benchmark suite
├── results/                   # Optimization outputs
├── README.md
└── requirements.txt
```

## Usage Examples

### 1. Define a Robot Morphology

```python
from morpho_compiler.src.morphodsl import MorphoChain, planar_leg_morphology

# Method 1: Fluent builder API
robot = (MorphoChain()
    .add_link(length=0.3, mass=0.5, radius=0.04)
    .add_joint(axis=[0, 1, 0], limits=(-1.5, 1.5))
    .add_link(length=0.3, mass=0.4, radius=0.03)
    .add_joint(axis=[0, 1, 0], limits=(-2.0, 0.5))
    .add_link(length=0.1, mass=0.2, radius=0.02)
    .build())

# Method 2: Use predefined templates
leg = planar_leg_morphology()

# Extract trainable parameters
params = robot.get_trainable_params()
print(params.keys())
# dict_keys(['link_0_length', 'link_0_mass', 'joint_0_damping', ...])
```

### 2. Run Physics Simulation

```python
from morpho_compiler.src.physics_engine import simulate_trajectory, compute_link_transforms
import jax.numpy as jnp

# Initial state
angles = jnp.zeros(2)  # 2 joints
velocities = jnp.zeros(2)

# Torque sequence (open-loop for demo)
torques = jnp.array([[0.5, -0.3]] * 50)  # 50 timesteps

# Run simulation
angle_traj, vel_traj = simulate_trajectory(
    angles, velocities, torques, robot, n_steps=50
)

# Get final link positions
final_angles = angle_traj[-1]
positions, orientations = compute_link_transforms(final_angles, robot)
print(f"Foot position: {positions[-1]}")
```

### 3. Validate Gradients

```python
from morpho_compiler.src.physics_engine import validate_gradients_finite_diff

def simple_loss(morph, ctrl):
    return sum(link.length[0]**2 for link in morph.links)

validation = validate_gradients_finite_diff(simple_loss, robot, {})

for param, result in validation.items():
    status = "✓" if result['passed'] else "✗"
    print(f"{status} {param}: error={result['relative_error']:.6f}")
```

### 4. Full Co-Optimization

```python
from morpho_compiler.src.main import MorphoCompiler

# Create compiler
compiler = MorphoCompiler(
    n_joints=4,
    sim_dt=0.01,
    sim_steps=100,
    output_dir="./results"
)

# Run optimization
results = compiler.optimize(n_iterations=100, log_interval=10)

# Access results
final_morphology = results['final_morphology']
history = results['history']

print(f"Final loss: {history['loss'][-1]:.4f}")
print(f"Optimization time: {results['total_time']:.1f}s")
```

### 5. Export for Fabrication

```python
from morpho_compiler.src.fabrication import MorphologyToCADConverter

converter = MorphologyToCADConverter(output_dir="./cad_output")

# Generate CAD components
components, instructions = converter.generate_assembly(final_morphology)

# Export files
converter.export_to_openscad(components, "optimized_robot")
converter.export_bom(components)
cost = converter.estimate_cost(components)

print(f"Estimated cost: ${cost['total_cost']:.2f}")
print(f"Print time: {cost['print_time_hours']:.1f} hours")
```

## Technical Details

### Differentiable Contact Model

The core technical challenge is making contact dynamics differentiable. We use a **soft contact penalty method**:

```python
@jit
def soft_contact_model(pos1, pos2, radius1, radius2, stiffness=1e4):
    delta = pos2 - pos1
    distance = jnp.linalg.norm(delta)
    penetration = jnp.maximum(0.0, radius1 + radius2 - distance)
    
    # Smooth penalty force (differentiable!)
    force = stiffness * penetration * (delta / distance)
    return force * (penetration > 0)
```

This allows gradients to flow through contact events, enabling morphology optimization.

### Multi-Objective Loss Function

```python
loss = -forward_progress      # Maximize velocity
       + 0.001 * energy       # Minimize power
       + 0.1 * instability    # Maintain balance
       + 10.0 * morph_reg     # Reasonable dimensions
```

## Benchmarks

### Locomotion Benchmark Suite

| Environment | Baseline (Fixed Morph) | Co-Optimized | Improvement |
|-------------|------------------------|--------------|-------------|
| Planar Walker | 0.45 m/s | TBD | - |
| Quadruped Trot | 0.62 m/s | TBD | - |
| Hopper | 1.2 m/s | TBD | - |

*Run benchmarks with: `python benchmarks/locomotion_benchmark.py`*

## Research Applications

This system enables novel research directions:

1. **Morphological Computation**: How much "intelligence" can be offloaded to body design?
2. **Evolutionary Robotics**: Gradient-based alternative to evolutionary algorithms
3. **Task-Specific Robots**: Automatically design robots optimized for specific environments
4. **Soft Robotics**: Discover non-trivial compliant mechanisms
5. **Sim-to-Real**: Study transfer of co-designed systems to physical hardware

## Known Limitations

1. **Simplified Physics**: Current implementation uses approximate dynamics (not full Rigid Body Algorithm)
2. **Contact Model**: Soft contact may not capture all real-world phenomena
3. **Compute Cost**: Full co-optimization requires significant GPU resources
4. **Fabrication Gap**: Continuous optimization → discrete parts needs manual refinement

## Roadmap

### Phase 1 (Current): Prototype ✓
- [x] Core differentiable physics
- [x] Basic morphology DSL
- [x] Gradient validation
- [ ] Full policy co-optimization

### Phase 2: Enhanced Capabilities
- [ ] Brax integration for faster simulation
- [ ] PPO/SAC policy training
- [ ] Multi-agent co-design
- [ ] Terrain adaptation

### Phase 3: Real-World Deployment
- [ ] Sim-to-real transfer protocols
- [ ] Hardware-in-the-loop optimization
- [ ] Automated fabrication pipeline
- [ ] ROS 2 integration

## Citation

If you use MorphoCompiler in your research, please cite:

```bibtex
@software{morphocompiler2024,
  title = {MorphoCompiler: Differentiable Robot Co-Design},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/morpho_compiler}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

This is a research project. Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Status**: Research-grade prototype suitable for academic collaboration, grant proposals, and early-stage development.

**Difficulty Level**: 9/10 - Requires expertise in JAX, robotics, and optimization theory.

**Timeline**: 6-9 months for publication-ready results.
