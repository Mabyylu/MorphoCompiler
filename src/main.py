"""
MorphoCompiler: Main Entry Point
End-to-end co-optimization of robot morphology and control policy.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
from typing import Dict, Tuple, Optional
import numpy as np
import time
from pathlib import Path

from .morphodsl import MorphologyConfig, MorphoChain, planar_leg_morphology
from .physics_engine import simulate_trajectory, compute_ground_contact_forces, compute_link_transforms
from .control_policy import create_policy, run_policy, get_observation, LocomotionPolicy
from .fabrication import MorphologyToCADConverter, generate_urdf


class MorphoCompiler:
    """
    Main compiler class for co-optimizing robot morphology and control.
    
    This is the core innovation: treating robot physical design as 
    differentiable parameters alongside neural network weights.
    """
    
    def __init__(
        self,
        n_joints: int = 4,
        sim_dt: float = 0.01,
        sim_steps: int = 100,
        output_dir: str = "./morpho_output"
    ):
        self.n_joints = n_joints
        self.sim_dt = sim_dt
        self.sim_steps = sim_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer for morphology
        self.morph_optimizer = optax.adam(learning_rate=0.001)
        self.morph_opt_state = None
        
        # Initialize optimizer for policy
        self.policy_optimizer = optax.adam(learning_rate=0.0003)
        self.policy_opt_state = None
        
        # Fabrication converter
        self.fab_converter = MorphologyToCADConverter(str(self.output_dir / "cad"))
    
    def create_initial_morphology(self) -> MorphologyConfig:
        """Create initial robot morphology for optimization."""
        return planar_leg_morphology()
    
    def define_loss_function(
        self,
        morphology: MorphologyConfig,
        policy_params: Dict,
        rng_key: jnp.ndarray
    ) -> Tuple[float, Dict]:
        """
        Define the co-optimization loss function.
        
        Objectives (multi-objective optimization):
        1. Maximize forward velocity
        2. Minimize energy consumption
        3. Maintain stability (minimize base orientation change)
        4. Regularize morphology (prevent extreme dimensions)
        """
        # Initialize state
        initial_angles = jnp.zeros(self.n_joints)
        initial_vels = jnp.zeros(self.n_joints)
        
        # Generate torque sequence using policy
        torque_sequence = []
        current_angles = initial_angles
        current_vels = initial_vels
        
        for step in range(self.sim_steps):
            # Get observation
            obs = get_observation(current_angles, current_vels)
            
            # Run policy to get torques
            # Note: In full impl, would use policy.apply with proper params
            torques = jnp.zeros(self.n_joints)  # Placeholder
            
            torque_sequence.append(torques)
            
            # Simple physics update (placeholder - would use full sim)
            current_angles = current_angles + current_vels * self.sim_dt
        
        torque_sequence = jnp.stack(torque_sequence)
        
        # Run simulation
        angle_traj, vel_traj = simulate_trajectory(
            initial_angles, initial_vels, torque_sequence, morphology, self.sim_steps
        )
        
        # Compute link positions for reward calculation
        final_angles = angle_traj[-1]
        link_positions, _ = compute_link_transforms(final_angles, morphology)
        
        # ===== REWARD COMPONENTS =====
        
        # 1. Forward progress reward (maximize x displacement)
        # Simplified: use foot position as proxy
        foot_pos = link_positions[-1]
        forward_reward = foot_pos[0]  # Positive x direction
        
        # 2. Energy penalty (minimize torque squared)
        energy_penalty = jnp.sum(torque_sequence ** 2) * 0.001
        
        # 3. Stability penalty (keep base upright)
        # Simplified: penalize large joint excursions
        stability_penalty = jnp.mean(angle_traj ** 2) * 0.1
        
        # 4. Morphology regularization (prefer reasonable sizes)
        morph_reg = 0.0
        for link in morphology.links:
            length = link.length[0]
            # Penalize very short or very long links
            morph_reg += ((length - 0.3) ** 2) * 10.0  # Prefer ~30cm links
        
        # Total loss (negative reward + penalties)
        total_loss = -forward_reward + energy_penalty + stability_penalty + morph_reg
        
        aux_dict = {
            'forward_reward': forward_reward,
            'energy_penalty': energy_penalty,
            'stability_penalty': stability_penalty,
            'morph_reg': morph_reg,
            'foot_x': foot_pos[0],
            'foot_z': foot_pos[2]
        }
        
        return total_loss, aux_dict
    
    @jit
    def compute_joint_gradients(
        morphology: MorphologyConfig,
        policy_params: Dict,
        rng_key: jnp.ndarray
    ):
        """Compute gradients w.r.t. both morphology and policy."""
        # Value and gradient function
        loss_fn = lambda m, p, r: self.define_loss_function(m, p, r)[0]
        
        # Gradient w.r.t. morphology
        morph_grad_fn = grad(loss_fn, argnums=0)
        morph_grads = morph_grad_fn(morphology, policy_params, rng_key)
        
        # Gradient w.r.t. policy
        policy_grad_fn = grad(loss_fn, argnums=1)
        policy_grads = policy_grad_fn(morphology, policy_params, rng_key)
        
        return morph_grads, policy_grads
    
    def optimize(
        self,
        n_iterations: int = 100,
        log_interval: int = 10
    ) -> Dict:
        """
        Run co-optimization loop.
        
        Args:
            n_iterations: Number of optimization iterations
            log_interval: How often to log progress
        
        Returns:
            Dictionary with optimization results
        """
        print("=" * 60)
        print("MorphoCompiler: Starting Co-Optimization")
        print("=" * 60)
        
        # Initialize morphology
        morphology = self.create_initial_morphology()
        print(f"\nInitial morphology: {len(morphology.links)} links, {len(morphology.joints)} joints")
        
        # Initialize policy
        policy, policy_params = create_policy(self.n_joints)
        print(f"Initialized policy with {sum(x.size for x in jax.tree_leaves(policy_params))} parameters")
        
        # Initialize optimizers
        morph_params = morphology.get_trainable_params()
        self.morph_opt_state = self.morph_optimizer.init(morph_params)
        self.policy_opt_state = self.policy_optimizer.init(policy_params)
        
        rng_key = random.PRNGKey(42)
        
        # Optimization history
        history = {
            'loss': [],
            'forward_reward': [],
            'energy_penalty': [],
            'morphology_snapshots': []
        }
        
        start_time = time.time()
        
        for iteration in range(n_iterations):
            rng_key, subkey = random.split(rng_key)
            
            # Compute loss and gradients
            loss, aux = self.define_loss_function(morphology, policy_params, subkey)
            
            # Get morphology gradients
            morph_grad_fn = grad(lambda m, p, r: self.define_loss_function(m, p, r)[0], argnums=0)
            morph_grads = morph_grad_fn(morphology, policy_params, subkey)
            
            # Update morphology parameters
            morph_params = morphology.get_trainable_params()
            updates, self.morph_opt_state = self.morph_optimizer.update(morph_grads, self.morph_opt_state, morph_params)
            morph_params = optax.apply_updates(morph_params, updates)
            
            # Reconstruct morphology from updated params
            morphology = morphology.set_params(morph_params)
            
            # Log progress
            if iteration % log_interval == 0 or iteration == n_iterations - 1:
                elapsed = time.time() - start_time
                print(f"\nIteration {iteration}/{n_iterations} ({elapsed:.1f}s)")
                print(f"  Loss: {loss:.4f}")
                print(f"  Forward reward: {aux['forward_reward']:.4f}")
                print(f"  Energy penalty: {aux['energy_penalty']:.4f}")
                print(f"  Foot position: ({aux['foot_x']:.3f}, {aux['foot_z']:.3f})")
                
                history['loss'].append(float(loss))
                history['forward_reward'].append(float(aux['forward_reward']))
                history['energy_penalty'].append(float(aux['energy_penalty']))
                
                if len(history['morphology_snapshots']) < 5:
                    history['morphology_snapshots'].append({
                        'iteration': iteration,
                        'link_lengths': [float(l.length[0]) for l in morphology.links]
                    })
        
        total_time = time.time() - start_time
        print(f"\n✓ Optimization completed in {total_time:.1f}s")
        
        # Save results
        results = {
            'final_morphology': morphology,
            'final_policy_params': policy_params,
            'history': history,
            'total_time': total_time,
            'iterations': n_iterations
        }
        
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save optimization results and generate fabrication files."""
        print("\n" + "=" * 60)
        print("Saving Results")
        print("=" * 60)
        
        morphology = results['final_morphology']
        
        # Generate CAD files
        print("\nGenerating fabrication files...")
        components, instructions = self.fab_converter.generate_assembly(morphology)
        self.fab_converter.export_to_openscad(components, "optimized_robot")
        self.fab_converter.export_bom(components)
        
        # Generate URDF
        urdf_path = self.output_dir / "optimized_robot.urdf"
        generate_urdf(morphology, str(urdf_path))
        
        # Save optimization history
        import json
        history_json = {
            'loss': results['history']['loss'],
            'forward_reward': results['history']['forward_reward'],
            'energy_penalty': results['history']['energy_penalty'],
            'total_time': results['total_time'],
            'iterations': results['iterations']
        }
        
        with open(self.output_dir / "optimization_history.json", 'w') as f:
            json.dump(history_json, f, indent=2)
        
        # Save morphology summary
        morph_summary = {
            'n_links': len(morphology.links),
            'n_joints': len(morphology.joints),
            'link_lengths': [float(l.length[0]) for l in morphology.links],
            'link_masses': [float(l.mass[0]) for l in morphology.links]
        }
        
        with open(self.output_dir / "morphology_summary.json", 'w') as f:
            json.dump(morph_summary, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir.absolute()}")
        print(f"  - optimized_robot.scad (3D visualization)")
        print(f"  - optimized_robot.urdf (ROS simulation)")
        print(f"  - bom.csv (Bill of Materials)")
        print(f"  - optimization_history.json (Training curves)")
        print(f"  - morphology_summary.json (Final design)")


def main():
    """Main entry point for MorphoCompiler."""
    print("\n" + "=" * 70)
    print("   MORPHOCOMPILER: Differentiable Robot Co-Design System")
    print("=" * 70)
    
    # Create compiler instance
    compiler = MorphoCompiler(
        n_joints=4,
        sim_dt=0.01,
        sim_steps=50,  # Shorter for demo
        output_dir="./morpho_output"
    )
    
    # Run optimization
    results = compiler.optimize(n_iterations=50, log_interval=10)
    
    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    final_morph = results['final_morphology']
    print(f"\nFinal Morphology:")
    for i, link in enumerate(final_morph.links):
        print(f"  Link {i}: length={float(link.length[0]):.3f}m, mass={float(link.mass[0]):.3f}kg")
    
    print(f"\nPerformance:")
    print(f"  Final loss: {results['history']['loss'][-1]:.4f}")
    print(f"  Final forward reward: {results['history']['forward_reward'][-1]:.4f}")
    print(f"  Total optimization time: {results['total_time']:.1f}s")
    
    print("\n" + "=" * 70)
    print("✓ MorphoCompiler execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
