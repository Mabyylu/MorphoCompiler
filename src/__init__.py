# MorphoCompiler Package

"""
MorphoCompiler: Differentiable Robot Co-Design System

A research-grade framework for simultaneous optimization of robot
morphology and control policies using gradient-based methods.
"""

from .src.morphodsl import (
    LinkParams,
    JointParams,
    MorphologyConfig,
    MorphoChain,
    planar_leg_morphology,
    quadruped_morphology
)

from .src.physics_engine import (
    soft_contact_model,
    ContactState,
    compute_link_transforms,
    compute_gravity_forces,
    compute_ground_contact_forces,
    forward_dynamics_step,
    simulate_trajectory,
    compute_morphology_gradients,
    validate_gradients_finite_diff
)

from .src.control_policy import (
    LocomotionPolicy,
    create_policy,
    run_policy,
    get_observation,
    CPGController
)

from .src.fabrication import (
    CADComponent,
    AssemblyInstruction,
    MorphologyToCADConverter,
    generate_urdf
)

from .src.main import MorphoCompiler

__version__ = '0.1.0'
__author__ = 'Research Team'
__all__ = [
    # DSL
    'LinkParams',
    'JointParams', 
    'MorphologyConfig',
    'MorphoChain',
    'planar_leg_morphology',
    'quadruped_morphology',
    
    # Physics
    'soft_contact_model',
    'ContactState',
    'compute_link_transforms',
    'compute_gravity_forces',
    'compute_ground_contact_forces',
    'forward_dynamics_step',
    'simulate_trajectory',
    'compute_morphology_gradients',
    'validate_gradients_finite_diff',
    
    # Control
    'LocomotionPolicy',
    'create_policy',
    'run_policy',
    'get_observation',
    'CPGController',
    
    # Fabrication
    'CADComponent',
    'AssemblyInstruction',
    'MorphologyToCADConverter',
    'generate_urdf',
    
    # Main
    'MorphoCompiler'
]
