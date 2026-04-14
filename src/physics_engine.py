"""
MorphoCompiler Physics Engine: Differentiable Contact Dynamics
Implements differentiable physics simulation for robot co-optimization.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from .morphodsl import MorphologyConfig, LinkParams, JointParams


@dataclass
class ContactState:
    """Differentiable contact state information."""
    penetration_depth: jnp.ndarray
    contact_normal: jnp.ndarray
    contact_point: jnp.ndarray
    friction_coeff: float = 0.5
    restitution: float = 0.3
    in_contact: bool = False


@jit
def soft_contact_model(
    pos1: jnp.ndarray, 
    pos2: jnp.ndarray, 
    radius1: float, 
    radius2: float,
    stiffness: float = 1e4,
    damping: float = 100.0
) -> Tuple[jnp.ndarray, ContactState]:
    """
    Differentiable soft contact model using penalty method.
    
    This is the core innovation enabling gradient-based morphology optimization.
    The contact force is a smooth function of penetration depth, allowing
    gradients to flow through contact events.
    
    Args:
        pos1, pos2: Positions of two contacting bodies (3D vectors)
        radius1, radius2: Effective radii of the bodies
        stiffness: Contact stiffness (spring constant)
        damping: Contact damping coefficient
    
    Returns:
        contact_force: 3D force vector applied to body 1
        contact_state: Contact information for analysis
    """
    # Relative position
    delta = pos2 - pos1
    distance = jnp.linalg.norm(delta) + 1e-8  # Avoid division by zero
    
    # Contact normal (direction from body 1 to body 2)
    normal = delta / distance
    
    # Penetration depth (positive when overlapping)
    penetration = jnp.maximum(0.0, radius1 + radius2 - distance)
    
    # Smooth penetration using softplus for better gradients near zero
    # penetration_smooth = jnp.log(1 + jnp.exp(penetration * 10)) / 10
    
    # Penalty force: spring-damper model
    # Elastic component (Hooke's law)
    elastic_force = stiffness * penetration * normal
    
    # Damping component (dissipative)
    # In full dynamics, this would use relative velocity
    damping_force = damping * penetration * normal
    
    # Total contact force (only when in contact)
    contact_mask = (penetration > 0).astype(jnp.float32)
    contact_force = (elastic_force + damping_force) * contact_mask
    
    # Create contact state
    contact_state = ContactState(
        penetration_depth=jnp.array([penetration]),
        contact_normal=normal,
        contact_point=pos1 + radius1 * normal,
        friction_coeff=0.5,
        restitution=0.3,
        in_contact=bool(penetration > 1e-6)
    )
    
    return contact_force, contact_state


@jit
def compute_link_transforms(
    joint_angles: jnp.ndarray,
    morphology: MorphologyConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute forward kinematics for serial chain robot.
    Fully differentiable w.r.t. both joint angles and morphology parameters.
    
    Args:
        joint_angles: Array of joint angles [n_joints]
        morphology: Robot morphology configuration
    
    Returns:
        link_positions: [n_links, 3] positions of link centers
        link_orientations: [n_links, 3, 3] rotation matrices
    """
    n_links = len(morphology.links)
    n_joints = len(morphology.joints)
    
    # Initialize arrays
    positions = jnp.zeros((n_links, 3))
    orientations = jnp.tile(jnp.eye(3), (n_links, 1, 1))
    
    current_pos = morphology.base_position
    current_rot = jnp.eye(3)
    
    # First link (base link)
    if n_links > 0:
        link_length = morphology.links[0].length[0]
        # Position at center of first link
        positions = positions.at[0].set(current_pos + jnp.array([0, 0, link_length/2]))
        orientations = orientations.at[0].set(current_rot)
    
    # Iterate through joints and subsequent links
    for i in range(min(n_joints, n_links - 1)):
        joint = morphology.joints[i]
        link_next = morphology.links[i + 1]
        
        # Get joint angle
        theta = joint_angles[i] if i < len(joint_angles) else 0.0
        
        # Rotation matrix from joint angle using Rodrigues' formula
        axis = joint.axis
        axis = axis / (jnp.linalg.norm(axis) + 1e-8)  # Normalize
        
        # Skew-symmetric matrix for cross product
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
        R_joint = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
        
        # Update orientation
        current_rot = current_rot @ R_joint
        
        # Move to next joint position
        link_current = morphology.links[i]
        link_dir = current_rot @ jnp.array([0, 0, 1])  # Local z-axis
        current_pos = current_pos + link_dir * link_current.length[0]
        
        # Position at center of next link
        link_length_next = link_next.length[0]
        positions = positions.at[i + 1].set(current_pos + link_dir * link_length_next / 2)
        orientations = orientations.at[i + 1].set(current_rot)
    
    return positions, orientations


@jit
def compute_gravity_forces(
    link_positions: jnp.ndarray,
    morphology: MorphologyConfig
) -> jnp.ndarray:
    """Compute gravity forces on all links."""
    g = jnp.array([0.0, 0.0, -9.81])
    forces = []
    for i, link in enumerate(morphology.links):
        mass = link.mass[0]
        forces.append(mass * g)
    return jnp.stack(forces)


@jit
def compute_ground_contact_forces(
    link_positions: jnp.ndarray,
    link_orientations: jnp.ndarray,
    morphology: MorphologyConfig,
    ground_height: float = 0.0
) -> jnp.ndarray:
    """
    Compute contact forces between robot links and ground plane.
    Uses differentiable soft contact model for each link.
    
    Args:
        link_positions: [n_links, 3] link center positions
        link_orientations: [n_links, 3, 3] link orientations
        morphology: Robot morphology
        ground_height: Z-coordinate of ground plane
    
    Returns:
        contact_forces: [n_links, 3] contact force on each link
    """
    n_links = len(morphology.links)
    contact_forces = jnp.zeros((n_links, 3))
    
    for i in range(n_links):
        pos = link_positions[i]
        link = morphology.links[i]
        
        # Model link as sphere for contact (simplified)
        radius = link.radius
        
        # Check if link is below ground height + radius
        # Ground contact point directly below link
        ground_pos = jnp.array([pos[0], pos[1], ground_height])
        
        # Compute contact with ground plane
        # For flat ground, we use a large "radius" approximation
        if pos[2] < ground_height + radius:
            # Contact with ground (ground has effectively infinite radius)
            force, _ = soft_contact_model(
                pos, ground_pos,
                radius, 0.0,
                stiffness=1e4,
                damping=50.0
            )
            contact_forces = contact_forces.at[i].add(force)
    
    return contact_forces


@jit
def forward_dynamics_step(
    joint_angles: jnp.ndarray,
    joint_velocities: jnp.ndarray,
    torques: jnp.ndarray,
    morphology: MorphologyConfig,
    dt: float = 0.01
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of forward dynamics simulation using semi-implicit Euler.
    
    This is the core simulation step that propagates the robot state forward
    in time given control inputs and physical parameters.
    
    Args:
        joint_angles: Current joint angles [n_joints]
        joint_velocities: Current joint velocities [n_joints]
        torques: Applied joint torques [n_joints]
        morphology: Robot morphology configuration
        dt: Time step size
    
    Returns:
        new_angles: Updated joint angles
        new_velocities: Updated joint velocities
    """
    n_joints = len(morphology.joints)
    n_links = len(morphology.links)
    
    # Compute link transforms for current state
    link_positions, link_orientations = compute_link_transforms(joint_angles, morphology)
    
    # Compute external forces
    gravity_forces = compute_gravity_forces(link_positions, morphology)
    contact_forces = compute_ground_contact_forces(link_positions, link_orientations, morphology)
    total_external_forces = gravity_forces + contact_forces
    
    # Simplified inverse dynamics to compute accelerations
    # In a full implementation, this would use the Articulated Body Algorithm
    # Here we use a simplified per-joint model
    
    accelerations = jnp.zeros(n_joints)
    
    for i in range(n_joints):
        joint = morphology.joints[i]
        
        # Control torque
        control_torque = torques[i] if i < len(torques) else 0.0
        
        # Joint damping torque
        damping_torque = -joint.damping * joint_velocities[i]
        
        # Gravity torque approximation
        # Sum effect of all downstream links
        gravity_torque = 0.0
        for j in range(i + 1, n_links):
            link = morphology.links[j]
            link_pos = link_positions[j]
            # Approximate torque from gravity acting on link
            moment_arm = link_pos[0]  # Simplified x-distance from joint
            gravity_torque -= link.mass[0] * 9.81 * moment_arm * jnp.cos(joint_angles[i] if i < len(joint_angles) else 0)
        
        # Net torque on joint
        net_torque = control_torque + damping_torque + gravity_torque
        
        # Simplified inertia calculation
        # Sum of downstream link inertias (very approximate)
        inertia = 0.1  # Base inertia
        for j in range(i + 1, n_links):
            link = morphology.links[j]
            inertia += link.mass[0] * link.length[0]**2
        
        # Angular acceleration
        accelerations = accelerations.at[i].set(net_torque / (inertia + 1e-6))
    
    # Semi-implicit Euler integration
    new_velocities = joint_velocities + accelerations * dt
    new_angles = joint_angles + new_velocities * dt
    
    # Enforce joint limits
    for i in range(n_joints):
        joint = morphology.joints[i]
        new_angles = new_angles.at[i].set(
            jnp.clip(new_angles[i], joint.lower_limit, joint.upper_limit)
        )
    
    return new_angles, new_velocities


@jit
def simulate_trajectory(
    initial_angles: jnp.ndarray,
    initial_velocities: jnp.ndarray,
    torque_sequence: jnp.ndarray,
    morphology: MorphologyConfig,
    n_steps: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate robot trajectory over time given torque commands.
    
    Uses JAX's lax.scan for efficient, differentiable simulation.
    
    Args:
        initial_angles: Starting joint angles [n_joints]
        initial_velocities: Starting joint velocities [n_joints]
        torque_sequence: [n_steps, n_joints] torque commands
        morphology: Robot morphology
        n_steps: Number of simulation steps
    
    Returns:
        angle_trajectory: [n_steps, n_joints] joint angle history
        velocity_trajectory: [n_steps, n_joints] joint velocity history
    """
    def step(carry, torques):
        angles, velocities = carry
        new_angles, new_velocities = forward_dynamics_step(
            angles, velocities, torques, morphology
        )
        return (new_angles, new_velocities), (new_angles, new_velocities)
    
    init_carry = (initial_angles, initial_velocities)
    _, (angle_traj, vel_traj) = jax.lax.scan(step, init_carry, torque_sequence)
    
    return angle_traj, vel_traj


# ============================================================================
# Gradient Computation for Co-Optimization
# ============================================================================

def compute_morphology_gradients(
    loss_fn,
    morphology: MorphologyConfig,
    controller_params: dict
) -> Dict[str, jnp.ndarray]:
    """
    Compute gradients of loss w.r.t. morphology parameters.
    
    This is the core capability enabling co-optimization of body and brain.
    Uses JAX's automatic differentiation through the entire simulation.
    
    Args:
        loss_fn: Function computing scalar loss from (morphology, controller)
        morphology: Current robot morphology
        controller_params: Controller parameters
    
    Returns:
        Dictionary of gradients for each morphology parameter
    """
    morph_params = morphology.get_trainable_params()
    
    def loss_wrapper(param_values_tuple):
        # Reconstruct param dict from tuple
        param_dict = dict(zip(morph_params.keys(), param_values_tuple))
        # Create updated morphology
        updated_morph = morphology.set_params(param_dict)
        # Compute loss
        return loss_fn(updated_morph, controller_params)
    
    # Compute gradients
    param_values = tuple(morph_params.values())
    grads_tuple = grad(loss_wrapper)(param_values)
    grads_dict = dict(zip(morph_params.keys(), grads_tuple))
    
    return grads_dict


def validate_gradients_finite_diff(
    loss_fn,
    morphology: MorphologyConfig,
    controller_params: dict,
    epsilon: float = 1e-5
) -> dict:
    """
    Validate analytical gradients against finite differences.
    
    Critical for ensuring the differentiable physics implementation is correct.
    
    Returns:
        Dictionary with validation results for each parameter
    """
    morph_params = morphology.get_trainable_params()
    
    # Compute analytical gradients
    try:
        analytical_grads = compute_morphology_gradients(loss_fn, morphology, controller_params)
    except Exception as e:
        return {'error': str(e)}
    
    numerical_grads = {}
    validation_results = {}
    
    for param_name, param_value in morph_params.items():
        # Compute numerical gradient via finite differences
        num_grad = jnp.zeros_like(param_value)
        
        for idx in np.ndindex(param_value.shape):
            # Perturb parameter positively
            param_plus = param_value.at[idx].add(epsilon)
            
            # Perturb parameter negatively  
            param_minus = param_value.at[idx].subtract(epsilon)
            
            # Create perturbed morphologies
            param_dict_plus = morph_params.copy()
            param_dict_plus[param_name] = param_plus
            morph_plus = morphology.set_params(param_dict_plus)
            
            param_dict_minus = morph_params.copy()
            param_dict_minus[param_name] = param_minus
            morph_minus = morphology.set_params(param_dict_minus)
            
            # Evaluate loss
            loss_plus = loss_fn(morph_plus, controller_params)
            loss_minus = loss_fn(morph_minus, controller_params)
            
            # Central difference
            num_grad = num_grad.at[idx].set((loss_plus - loss_minus) / (2 * epsilon))
        
        numerical_grads[param_name] = num_grad
        
        # Compare analytical vs numerical
        anal_grad = analytical_grads.get(param_name, jnp.zeros_like(param_value))
        
        # Relative error metric
        diff_norm = jnp.linalg.norm(anal_grad - num_grad)
        scale = jnp.linalg.norm(anal_grad) + jnp.linalg.norm(num_grad) + 1e-8
        relative_error = diff_norm / scale
        
        validation_results[param_name] = {
            'analytical_norm': float(jnp.linalg.norm(anal_grad)),
            'numerical_norm': float(jnp.linalg.norm(num_grad)),
            'relative_error': float(relative_error),
            'passed': bool(relative_error < 0.01)  # 1% tolerance
        }
    
    return validation_results


if __name__ == "__main__":
    print("=" * 60)
    print("MorphoCompiler Physics Engine: Differentiable Contact Dynamics")
    print("=" * 60)
    
    from .morphodsl import MorphoChain, planar_leg_morphology
    
    # Test 1: Soft contact model
    print("\n1. Testing soft contact model...")
    pos1 = jnp.array([0.0, 0.0, 0.05])  # 5cm above ground
    pos2 = jnp.array([0.0, 0.0, 0.0])   # Ground level
    force, state = soft_contact_model(pos1, pos2, 0.06, 0.0, stiffness=1e4)
    print(f"   Position: {pos1[2]:.3f}m, Radius: 0.06m")
    print(f"   Penetration: {state.penetration_depth[0]:.4f}m")
    print(f"   Contact force: {force[2]:.2f}N (upward)")
    assert state.in_contact, "Should be in contact"
    assert force[2] > 0, "Force should be upward"
    
    # Test 2: Forward kinematics
    print("\n2. Testing forward kinematics...")
    config = planar_leg_morphology()
    joint_angles = jnp.array([0.3, -0.6])  # Hip and knee angles
    positions, orientations = compute_link_transforms(joint_angles, config)
    print(f"   Created {len(config.links)}-link leg")
    print(f"   Foot position: ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})")
    
    # Test 3: Dynamics simulation
    print("\n3. Testing forward dynamics...")
    initial_angles = jnp.array([0.0, 0.0])
    initial_vels = jnp.array([0.0, 0.0])
    torques = jnp.array([[0.5, -0.3]] * 10)  # 10 steps of constant torque
    angle_traj, vel_traj = simulate_trajectory(
        initial_angles, initial_vels, torques, config, n_steps=10
    )
    print(f"   Simulated {len(angle_traj)} steps")
    print(f"   Final angles: ({angle_traj[-1, 0]:.3f}, {angle_traj[-1, 1]:.3f}) rad")
    
    # Test 4: Gradient computation
    print("\n4. Testing gradient computation...")
    def simple_loss(morph, ctrl):
        """Simple loss: minimize total link length."""
        total_length = sum(link.length[0] for link in morph.links)
        return total_length
    
    grads = compute_morphology_gradients(simple_loss, config, {})
    print(f"   Gradient w.r.t. link_0_length: {grads['link_0_length']}")
    print(f"   Gradient w.r.t. link_1_length: {grads['link_1_length']}")
    
    # Test 5: Gradient validation
    print("\n5. Validating gradients with finite differences...")
    validation = validate_gradients_finite_diff(simple_loss, config, {})
    all_passed = True
    for param_name, result in validation.items():
        status = "✓" if result.get('passed', False) else "✗"
        print(f"   {status} {param_name}: rel_error={result.get('relative_error', 'N/A'):.6f}")
        if not result.get('passed', False):
            all_passed = False
    
    if all_passed:
        print("\n✓ All physics engine tests passed!")
    else:
        print("\n⚠ Some gradient validations failed (may need tuning)")
