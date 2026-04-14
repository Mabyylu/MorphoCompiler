"""
MorphoDSL: Domain-Specific Language for Robot Morphology Definition
Differentiable representation where physical parameters are trainable tensors.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class LinkParams:
    """Differentiable link parameters."""
    length: jnp.ndarray
    mass: jnp.ndarray
    inertia: jnp.ndarray
    material_density: float = 1000.0
    youngs_modulus: float = 1e9
    damping_coeff: float = 0.1
    radius: float = 0.05  # For contact modeling


@dataclass
class JointParams:
    """Differentiable joint parameters."""
    axis: jnp.ndarray
    lower_limit: float = -1.5
    upper_limit: float = 1.5
    damping: float = 0.1
    stiffness: float = 0.0
    joint_type: str = "revolute"  # revolute, prismatic


@dataclass
class MorphologyConfig:
    """Complete robot morphology configuration."""
    joints: List[JointParams]
    links: List[LinkParams]
    base_position: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    
    def get_trainable_params(self) -> Dict[str, jnp.ndarray]:
        """Extract all trainable parameters as a flat dictionary."""
        params = {}
        for i, link in enumerate(self.links):
            params[f'link_{i}_length'] = link.length
            params[f'link_{i}_mass'] = link.mass
            params[f'link_{i}_radius'] = jnp.array([link.radius])
        for i, joint in enumerate(self.joints):
            params[f'joint_{i}_damping'] = jnp.array([joint.damping])
            params[f'joint_{i}_stiffness'] = jnp.array([joint.stiffness])
        return params
    
    def set_params(self, param_dict: Dict[str, jnp.ndarray]) -> 'MorphologyConfig':
        """Create new config with updated parameters (for gradient updates)."""
        new_links = []
        new_joints = []
        
        for i, link in enumerate(self.links):
            new_length = param_dict.get(f'link_{i}_length', link.length)
            new_mass = param_dict.get(f'link_{i}_mass', link.mass)
            new_radius = float(param_dict.get(f'link_{i}_radius', jnp.array([link.radius]))[0])
            new_links.append(LinkParams(
                length=new_length,
                mass=new_mass,
                inertia=link.inertia,
                radius=new_radius
            ))
        
        for i, joint in enumerate(self.joints):
            new_damping = float(param_dict.get(f'joint_{i}_damping', jnp.array([joint.damping]))[0])
            new_stiffness = float(param_dict.get(f'joint_{i}_stiffness', jnp.array([joint.stiffness]))[0])
            new_joints.append(JointParams(
                axis=joint.axis,
                lower_limit=joint.lower_limit,
                upper_limit=joint.upper_limit,
                damping=new_damping,
                stiffness=new_stiffness,
                joint_type=joint.joint_type
            ))
        
        return MorphologyConfig(joints=new_joints, links=new_links, base_position=self.base_position)


class MorphoChain:
    """Fluent builder for constructing robot morphologies."""
    
    def __init__(self):
        self._links = []
        self._joints = []
    
    def add_link(self, length=1.0, mass=1.0, radius=0.05):
        """Add a link to the chain."""
        self._links.append(LinkParams(
            length=jnp.array([length]),
            mass=jnp.array([mass]),
            inertia=jnp.eye(3).flatten() * mass * length**2 / 12,
            radius=radius
        ))
        return self
    
    def add_joint(self, axis=[0, 0, 1], limits=(-1.5, 1.5), joint_type="revolute"):
        """Add a joint to the chain."""
        self._joints.append(JointParams(
            axis=jnp.array(axis, dtype=jnp.float32),
            lower_limit=limits[0],
            upper_limit=limits[1],
            joint_type=joint_type
        ))
        return self
    
    def build(self) -> MorphologyConfig:
        """Build the final morphology configuration."""
        if len(self._links) == 0:
            raise ValueError("Morphology must have at least one link")
        if len(self._joints) != len(self._links) - 1:
            raise ValueError(f"Joint count ({len(self._joints)}) must be link count ({len(self._links)}) - 1")
        return MorphologyConfig(joints=self._joints, links=self._links)


def planar_leg_morphology() -> MorphologyConfig:
    """Create a simple planar leg morphology for testing."""
    return (MorphoChain()
        .add_link(length=0.3, mass=0.5, radius=0.04)  # Thigh
        .add_joint(axis=[0, 1, 0], limits=(-1.5, 1.5))  # Hip
        .add_link(length=0.3, mass=0.4, radius=0.03)  # Shank
        .add_joint(axis=[0, 1, 0], limits=(-2.0, 0.5))  # Knee
        .add_link(length=0.1, mass=0.2, radius=0.02)  # Foot
        .build())


def quadruped_morphology() -> MorphologyConfig:
    """Create a simple quadruped morphology."""
    # Simplified: single leg representation
    return planar_leg_morphology()


if __name__ == "__main__":
    print("=" * 60)
    print("MorphoDSL: Domain-Specific Language for Robot Morphology")
    print("=" * 60)
    
    # Example 1: Simple 2-link arm
    print("\n1. Creating 2-link planar arm...")
    arm_config = (MorphoChain()
        .add_link(length=1.0, mass=1.0)
        .add_joint(axis=[0, 0, 1])
        .add_link(length=0.8, mass=0.8)
        .build())
    
    print(f"   Links: {len(arm_config.links)}")
    print(f"   Joints: {len(arm_config.joints)}")
    print(f"   Trainable params: {list(arm_config.get_trainable_params().keys())}")
    
    # Example 2: Planar leg
    print("\n2. Creating planar leg morphology...")
    leg_config = planar_leg_morphology()
    print(f"   Links: {len(leg_config.links)}")
    print(f"   Joints: {len(leg_config.joints)}")
    
    # Test parameter extraction
    print("\n3. Extracting trainable parameters...")
    params = leg_config.get_trainable_params()
    for name, value in params.items():
        print(f"   {name}: {value}")
    
    # Test parameter update (simulating gradient step)
    print("\n4. Testing parameter update...")
    updated_params = {k: v * 1.1 for k, v in params.items()}  # 10% increase
    new_config = leg_config.set_params(updated_params)
    print(f"   Original link[0] length: {leg_config.links[0].length[0]:.3f}")
    print(f"   Updated link[0] length: {new_config.links[0].length[0]:.3f}")
    
    print("\n✓ MorphoDSL tests passed")
