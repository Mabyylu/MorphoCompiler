"""Unit tests for physics engine module."""

import jax.numpy as jnp
import pytest
from jax import grad
from morpho_compiler.src.physics_engine import (
    soft_contact_model,
    compute_link_transforms,
    compute_gravity_forces,
    forward_dynamics_step,
    simulate_trajectory
)
from morpho_compiler.src.morphodsl import MorphoChain, planar_leg_morphology


class TestSoftContactModel:
    """Test differentiable contact model."""
    
    def test_no_contact(self):
        """Test when bodies are not in contact."""
        pos1 = jnp.array([0.0, 0.0, 0.2])
        pos2 = jnp.array([0.0, 0.0, 0.0])
        force, state = soft_contact_model(pos1, pos2, 0.05, 0.0)
        
        assert not state.in_contact
        assert jnp.all(force == 0)
    
    def test_in_contact(self):
        """Test when bodies are in contact."""
        pos1 = jnp.array([0.0, 0.0, 0.05])
        pos2 = jnp.array([0.0, 0.0, 0.0])
        force, state = soft_contact_model(pos1, pos2, 0.06, 0.0)
        
        assert state.in_contact
        assert state.penetration_depth[0] > 0
        assert force[2] > 0  # Upward force
    
    def test_contact_gradient(self):
        """Test that contact force is differentiable."""
        pos1 = jnp.array([0.0, 0.0, 0.05])
        pos2 = jnp.array([0.0, 0.0, 0.0])
        
        def force_fn(z):
            p1 = jnp.array([0.0, 0.0, z])
            f, _ = soft_contact_model(p1, pos2, 0.06, 0.0)
            return f[2]
        
        # Should be able to compute gradient
        grad_fn = grad(force_fn)
        g = grad_fn(0.05)
        assert jnp.isfinite(g)


class TestForwardKinematics:
    """Test forward kinematics computation."""
    
    def test_zero_angles(self):
        """Test with zero joint angles."""
        config = planar_leg_morphology()
        angles = jnp.zeros(2)
        
        positions, orientations = compute_link_transforms(angles, config)
        
        # First link should be at base
        assert positions[0, 2] > 0  # Above ground
        
    def test_bent_leg(self):
        """Test with bent leg configuration."""
        config = planar_leg_morphology()
        angles = jnp.array([0.5, -0.8])  # Hip and knee bent
        
        positions, _ = compute_link_transforms(angles, config)
        
        # Foot should move forward when leg bends
        assert positions[-1, 0] > 0  # Positive x displacement


class TestDynamics:
    """Test dynamics simulation."""
    
    def test_gravity_only(self):
        """Test falling under gravity."""
        config = planar_leg_morphology()
        angles = jnp.zeros(2)
        velocities = jnp.zeros(2)
        torques = jnp.zeros(2)
        
        new_angles, new_vels = forward_dynamics_step(
            angles, velocities, torques, config, dt=0.01
        )
        
        # Velocities should become negative (falling)
        assert jnp.any(new_vels < 0)
    
    def test_apply_torque(self):
        """Test applying joint torque."""
        config = planar_leg_morphology()
        angles = jnp.zeros(2)
        velocities = jnp.zeros(2)
        torques = jnp.array([1.0, -1.0])
        
        new_angles, new_vels = forward_dynamics_step(
            angles, velocities, torques, config, dt=0.01
        )
        
        # Velocities should reflect applied torque
        assert new_vels[0] > 0  # Positive torque → positive velocity
        assert new_vels[1] < 0  # Negative torque → negative velocity
    
    def test_trajectory_simulation(self):
        """Test multi-step simulation."""
        config = planar_leg_morphology()
        angles = jnp.zeros(2)
        velocities = jnp.zeros(2)
        torques = jnp.array([[0.5, -0.3]] * 20)
        
        angle_traj, vel_traj = simulate_trajectory(
            angles, velocities, torques, config, n_steps=20
        )
        
        assert angle_traj.shape == (20, 2)
        assert vel_traj.shape == (20, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
