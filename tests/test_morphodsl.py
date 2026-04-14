"""Unit tests for MorphoDSL module."""

import jax.numpy as jnp
import pytest
from morpho_compiler.src.morphodsl import (
    LinkParams, JointParams, MorphologyConfig,
    MorphoChain, planar_leg_morphology
)


class TestLinkParams:
    """Test LinkParams dataclass."""
    
    def test_create_link(self):
        link = LinkParams(
            length=jnp.array([1.0]),
            mass=jnp.array([2.0]),
            inertia=jnp.ones(9)
        )
        assert float(link.length[0]) == 1.0
        assert float(link.mass[0]) == 2.0
    
    def test_default_values(self):
        link = LinkParams(
            length=jnp.array([1.0]),
            mass=jnp.array([1.0]),
            inertia=jnp.ones(9)
        )
        assert link.material_density == 1000.0
        assert link.youngs_modulus == 1e9
        assert link.radius == 0.05


class TestMorphoChain:
    """Test MorphoChain builder."""
    
    def test_build_simple_chain(self):
        chain = (MorphoChain()
            .add_link(length=1.0, mass=1.0)
            .add_joint(axis=[0, 0, 1])
            .add_link(length=0.8, mass=0.8)
            .build())
        
        assert len(chain.links) == 2
        assert len(chain.joints) == 1
    
    def test_planar_leg(self):
        leg = planar_leg_morphology()
        assert len(leg.links) == 3
        assert len(leg.joints) == 2
    
    def test_invalid_chain_no_links(self):
        with pytest.raises(ValueError):
            MorphoChain().build()
    
    def test_invalid_chain_mismatched_joints(self):
        with pytest.raises(ValueError):
            (MorphoChain()
                .add_link(length=1.0)
                .add_link(length=1.0)
                .build())  # Missing joint


class TestMorphologyConfig:
    """Test MorphologyConfig parameter handling."""
    
    def test_get_trainable_params(self):
        config = planar_leg_morphology()
        params = config.get_trainable_params()
        
        assert 'link_0_length' in params
        assert 'link_0_mass' in params
        assert 'joint_0_damping' in params
    
    def test_set_params(self):
        config = planar_leg_morphology()
        original_length = float(config.links[0].length[0])
        
        # Update parameters
        params = config.get_trainable_params()
        params['link_0_length'] = jnp.array([original_length * 1.5])
        
        new_config = config.set_params(params)
        new_length = float(new_config.links[0].length[0])
        
        assert abs(new_length - original_length * 1.5) < 1e-6
        # Original should be unchanged
        assert abs(float(config.links[0].length[0]) - original_length) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
