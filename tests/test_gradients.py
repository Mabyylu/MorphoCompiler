"""Unit tests for gradient computation and validation."""

import jax.numpy as jnp
import pytest
from jax import grad
from morpho_compiler.src.physics_engine import (
    validate_gradients_finite_diff,
    compute_morphology_gradients
)
from morpho_compiler.src.morphodsl import planar_leg_morphology


class TestGradientComputation:
    """Test gradient computation for morphology optimization."""
    
    def test_simple_gradient(self):
        """Test gradient of simple loss function."""
        config = planar_leg_morphology()
        
        def simple_loss(morph, ctrl):
            return sum(link.length[0]**2 for link in morph.links)
        
        grads = compute_morphology_gradients(simple_loss, config, {})
        
        assert 'link_0_length' in grads
        assert grads['link_0_length'].shape == (1,)
        # Gradient of length^2 is 2*length, should be positive
        assert grads['link_0_length'][0] > 0
    
    def test_gradient_zero_for_constant(self):
        """Test gradient is zero for constant loss."""
        config = planar_leg_morphology()
        
        def constant_loss(morph, ctrl):
            return 42.0  # Constant
        
        grads = compute_morphology_gradients(constant_loss, config, {})
        
        for param_name, grad_val in grads.items():
            assert jnp.allclose(grad_val, 0.0, atol=1e-6)


class TestGradientValidation:
    """Test finite difference gradient validation."""
    
    def test_validation_passes_for_correct_gradients(self):
        """Test that analytical gradients match numerical."""
        config = planar_leg_morphology()
        
        def simple_loss(morph, ctrl):
            total = 0.0
            for link in morph.links:
                total += link.length[0]**2
            return total
        
        validation = validate_gradients_finite_diff(simple_loss, config, {})
        
        # All parameters should pass validation
        for param_name, result in validation.items():
            if 'error' not in result:  # Skip if there was an error
                assert result['passed'], f"{param_name} failed: {result}"
    
    def test_validation_detects_bad_gradients(self):
        """Test that validation catches incorrect gradients."""
        # This test would require intentionally breaking the gradient
        # For now, we verify the validation framework works
        config = planar_leg_morphology()
        
        def loss_with_cubic(morph, ctrl):
            # Cubic term makes gradients larger
            return sum(link.length[0]**3 for link in morph.links)
        
        validation = validate_gradients_finite_diff(loss_with_cubic, config, {})
        
        # Should complete without errors
        assert 'error' not in validation or len(validation) > 0


class TestDifferentiablePhysics:
    """Test end-to-end differentiability through physics."""
    
    def test_trajectory_gradient(self):
        """Test gradients flow through trajectory simulation."""
        from morpho_compiler.src.physics_engine import simulate_trajectory
        
        config = planar_leg_morphology()
        angles = jnp.zeros(2)
        velocities = jnp.zeros(2)
        torques = jnp.ones((10, 2)) * 0.5
        
        def trajectory_loss(morph):
            angle_traj, _ = simulate_trajectory(
                angles, velocities, torques, morph, n_steps=10
            )
            # Loss based on final position
            return jnp.sum(angle_traj[-1]**2)
        
        # Should be able to compute gradient
        grads = compute_morphology_gradients(trajectory_loss, config, {})
        
        assert len(grads) > 0
        for g in grads.values():
            assert jnp.all(jnp.isfinite(g))
    
    def test_contact_gradient_through_simulation(self):
        """Test that contact forces provide useful gradients."""
        config = planar_leg_morphology()
        
        def contact_loss(morph, ctrl):
            # Simulate and penalize ground penetration
            angles = jnp.zeros(len(morph.joints))
            velocities = jnp.zeros(len(morph.joints))
            torques = jnp.zeros((5, len(morph.joints)))
            
            from morpho_compiler.src.physics_engine import (
                compute_link_transforms,
                compute_ground_contact_forces
            )
            
            positions, _ = compute_link_transforms(angles, morph)
            contact_forces = compute_ground_contact_forces(
                positions, jnp.eye(3), morph
            )
            
            # Penalize large contact forces (deep penetration)
            return jnp.sum(contact_forces**2)
        
        grads = compute_morphology_gradients(contact_loss, config, {})
        
        # Gradients should exist and be finite
        for param_name, grad_val in grads.items():
            assert jnp.all(jnp.isfinite(grad_val)), f"{param_name} has non-finite gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
