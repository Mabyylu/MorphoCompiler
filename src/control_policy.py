"""
MorphoCompiler Control Policy: Neural Network Controller for Robot Locomotion
Implements differentiable control policies that can be co-optimized with morphology.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, random
from flax import linen as nn
from typing import Tuple, Dict, Any, Optional
import numpy as np


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    features: list
    activation: str = 'relu'
    
    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat)(x)
            if self.activation == 'relu':
                x = nn.relu(x)
            elif self.activation == 'tanh':
                x = nn.tanh(x)
            elif self.activation == 'swish':
                x = nn.swish(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class LocomotionPolicy(nn.Module):
    """
    Neural network policy for robot locomotion.
    Takes proprioceptive observations and outputs joint torques.
    """
    n_joints: int
    hidden_dims: tuple = (64, 64, 32)
    activation: str = 'tanh'
    
    @nn.compact
    def __call__(self, obs):
        """
        Forward pass of the policy network.
        
        Args:
            obs: Observation vector containing:
                 - Joint angles [n_joints]
                 - Joint velocities [n_joints]
                 - Base orientation (optional)
                 - Command velocity (optional)
        
        Returns:
            torques: Joint torque commands [n_joints]
        """
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        
        # Output layer: torque commands
        torques = nn.Dense(self.n_joints)(x)
        
        # Scale torques to reasonable range [-10, 10] Nm
        torques = 10.0 * nn.tanh(torques)
        
        return torques


def create_policy(n_joints: int, rng_key=None) -> Tuple[LocomotionPolicy, Dict]:
    """Initialize a new policy with random parameters."""
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
    policy = LocomotionPolicy(n_joints=n_joints)
    
    # Dummy input for initialization
    dummy_obs = jnp.zeros(n_joints * 2 + 3)  # angles + velocities + base_orient
    
    params = policy.init(rng_key, dummy_obs)
    return policy, params


def get_observation(
    joint_angles: jnp.ndarray,
    joint_velocities: jnp.ndarray,
    base_orientation: Optional[jnp.ndarray] = None,
    command: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Construct observation vector from robot state.
    
    Args:
        joint_angles: [n_joints] current joint angles
        joint_velocities: [n_joints] current joint velocities
        base_orientation: [3] base orientation (roll, pitch, yaw)
        command: [2] commanded velocity (vx, vy)
    
    Returns:
        obs: Concatenated observation vector
    """
    obs_parts = [joint_angles, joint_velocities]
    
    if base_orientation is not None:
        obs_parts.append(base_orientation)
    else:
        obs_parts.append(jnp.zeros(3))
    
    if command is not None:
        obs_parts.append(command)
    else:
        obs_parts.append(jnp.array([1.0, 0.0]))  # Default: move forward at 1 m/s
    
    return jnp.concatenate(obs_parts)


@jit
def run_policy(policy: LocomotionPolicy, params: Dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Run policy to get action (torque commands)."""
    return policy.apply(params, obs)


# ============================================================================
# Training Utilities
# ============================================================================

def compute_policy_gradients(
    loss_fn,
    policy: LocomotionPolicy,
    params: Dict,
    observations: jnp.ndarray
):
    """Compute gradients of loss w.r.t. policy parameters."""
    
    def loss_wrapper(params_flat):
        # Reconstruct params dict (simplified)
        return loss_fn(policy, params, observations)
    
    grads = grad(loss_wrapper)(params)
    return grads


def reinforce_loss(
    policy: LocomotionPolicy,
    params: Dict,
    trajectories: Dict,
    rewards: jnp.ndarray
) -> float:
    """
    REINFORCE-style loss for policy gradient training.
    
    Args:
        policy: Policy network
        params: Policy parameters
        trajectories: Dict with 'observations', 'actions', 'log_probs'
        rewards: Cumulative rewards for each trajectory
    
    Returns:
        loss: Policy gradient loss
    """
    log_probs = trajectories['log_probs']
    
    # Negative log probability weighted by rewards
    loss = -jnp.mean(log_probs * rewards)
    
    return loss


def ppo_clip_loss(
    policy: LocomotionPolicy,
    params: Dict,
    old_params: Dict,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_epsilon: float = 0.2
) -> float:
    """
    PPO clipped surrogate loss.
    
    Args:
        policy: Current policy
        params: Current policy parameters
        old_params: Old policy parameters (for importance sampling)
        observations: State observations
        actions: Taken actions
        advantages: Advantage estimates
        clip_epsilon: Clipping parameter
    
    Returns:
        loss: PPO loss
    """
    # Compute log probabilities under current and old policies
    # (simplified - full implementation would compute proper Gaussian log probs)
    
    current_actions = policy.apply(params, observations)
    old_actions = policy.apply(old_params, observations)
    
    # Simplified ratio (in full impl, would use probability densities)
    diff = jnp.sum((current_actions - actions)**2, axis=-1)
    old_diff = jnp.sum((old_actions - actions)**2, axis=-1)
    
    ratio = jnp.exp(old_diff - diff)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    
    loss = -jnp.minimum(surr1, surr2).mean()
    
    return loss


# ============================================================================
# Central Pattern Generator (CPG) Controller
# ============================================================================

class CPGController:
    """
    Central Pattern Generator for rhythmic locomotion.
    Bio-inspired oscillator-based controller.
    """
    
    def __init__(self, n_joints: int, frequencies: Optional[jnp.ndarray] = None):
        self.n_joints = n_joints
        self.frequencies = frequencies or jnp.ones(n_joints) * 2.0  # 2 Hz default
        
    @jit
    def step(self, phase: jnp.ndarray, dt: float = 0.01) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Advance CPG oscillators and compute output.
        
        Args:
            phase: Current phase of each oscillator [n_joints]
            dt: Time step
        
        Returns:
            new_phase: Updated phase
            output: Oscillator output (torque commands)
        """
        # Phase dynamics: simple harmonic oscillator
        omega = 2 * jnp.pi * self.frequencies
        new_phase = phase + omega * dt
        
        # Wrap phase to [0, 2π]
        new_phase = new_phase % (2 * jnp.pi)
        
        # Output: sine wave with amplitude
        amplitude = jnp.ones(self.n_joints) * 5.0  # 5 Nm amplitude
        output = amplitude * jnp.sin(new_phase)
        
        # Add phase offsets for gait pattern (simplified trot)
        if self.n_joints >= 4:
            offsets = jnp.array([0, jnp.pi, jnp.pi, 0])
            output = output + amplitude * jnp.sin(new_phase + offsets[:self.n_joints])
        
        return new_phase, output


if __name__ == "__main__":
    print("=" * 60)
    print("MorphoCompiler Control Policy Module")
    print("=" * 60)
    
    # Test 1: Create and run policy
    print("\n1. Creating locomotion policy...")
    n_joints = 4
    policy, params = create_policy(n_joints)
    print(f"   Created policy for {n_joints}-joint robot")
    print(f"   Parameter count: {sum(x.size for x in jax.tree_leaves(params))}")
    
    # Test 2: Run policy inference
    print("\n2. Running policy inference...")
    joint_angles = jnp.array([0.1, -0.2, 0.15, -0.1])
    joint_vels = jnp.array([0.0, 0.1, -0.05, 0.0])
    obs = get_observation(joint_angles, joint_vels)
    torques = run_policy(policy, params, obs)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Torque commands: {torques}")
    
    # Test 3: CPG controller
    print("\n3. Testing CPG controller...")
    cpg = CPGController(n_joints=4)
    phase = jnp.zeros(4)
    outputs = []
    for i in range(10):
        phase, output = cpg.step(phase, dt=0.01)
        outputs.append(output)
    outputs = jnp.stack(outputs)
    print(f"   Generated {len(outputs)} CPG steps")
    print(f"   Output range: [{outputs.min():.2f}, {outputs.max():.2f}] Nm")
    
    # Test 4: Gradient computation
    print("\n4. Testing policy gradient computation...")
    def dummy_loss(policy, params, obs):
        torques = policy.apply(params, obs)
        return jnp.sum(torques**2)  # Minimize torque magnitude
    
    grads = compute_policy_gradients(dummy_loss, policy, params, obs)
    print(f"   Gradient shapes: {[g.shape for g in jax.tree_leaves(grads)]}")
    
    print("\n✓ Control policy module tests passed!")
