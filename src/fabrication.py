"""
MorphoCompiler Fabrication Module: Morphology-to-CAD Pipeline
Converts optimized differentiable morphologies to manufacturable CAD models.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class CADComponent:
    """Represents a single manufacturable component."""
    name: str
    geometry_type: str  # cylinder, box, sphere, custom
    dimensions: Dict[str, float]  # length, width, height, radius, etc.
    material: str
    mass: float
    connection_points: List[Dict[str, float]]  # xyz coordinates
    joint_type: Optional[str] = None
    tolerance: float = 0.1  # mm


@dataclass  
class AssemblyInstruction:
    """Assembly instruction for connecting components."""
    component_a: str
    component_b: str
    connection_type: str  # bolt, hinge, weld
    fastener_specs: Optional[Dict] = None
    torque_spec: Optional[float] = None


class MorphologyToCADConverter:
    """
    Converts optimized morphology parameters to CAD-ready components.
    
    This module handles the critical sim-to-fab transition, converting
    continuous differentiable parameters to discrete manufacturable parts.
    """
    
    def __init__(self, output_dir: str = "./cad_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard materials with properties
        self.materials = {
            'pla': {'density': 1240, 'youngs_modulus': 3.5e9},
            'abs': {'density': 1040, 'youngs_modulus': 2.3e9},
            'nylon': {'density': 1140, 'youngs_modulus': 2.7e9},
            'carbon_fiber': {'density': 1600, 'youngs_modulus': 70e9},
            'aluminum': {'density': 2700, 'youngs_modulus': 69e9}
        }
    
    def convert_link_to_component(
        self,
        link_params,
        link_id: int,
        material: str = 'carbon_fiber'
    ) -> CADComponent:
        """
        Convert a differentiable link parameter to CAD component.
        
        Handles the continuous-to-discrete conversion by:
        1. Rounding dimensions to standard sizes
        2. Adding manufacturing tolerances
        3. Generating connection interfaces
        """
        length = float(link_params.length[0])
        mass = float(link_params.mass[0])
        radius = link_params.radius
        
        # Round to nearest standard size (cm precision)
        length_rounded = round(length * 100) / 100  # Round to cm
        
        # Calculate connection points (ends of link)
        connection_points = [
            {'x': 0.0, 'y': 0.0, 'z': 0.0},  # Proximal end
            {'x': 0.0, 'y': 0.0, 'z': length_rounded}  # Distal end
        ]
        
        return CADComponent(
            name=f"link_{link_id}",
            geometry_type='cylinder',
            dimensions={
                'length': length_rounded,
                'radius': radius,
                'wall_thickness': max(radius * 0.2, 0.002)  # Min 2mm walls
            },
            material=material,
            mass=mass,
            connection_points=connection_points,
            tolerance=0.1
        )
    
    def convert_joint_to_component(
        self,
        joint_params,
        joint_id: int,
        material: str = 'aluminum'
    ) -> CADComponent:
        """Convert joint parameters to CAD component."""
        axis = joint_params.axis
        limits = (joint_params.lower_limit, joint_params.upper_limit)
        
        return CADComponent(
            name=f"joint_{joint_id}",
            geometry_type='custom',
            dimensions={
                'shaft_diameter': 0.008,  # 8mm standard
                'bearing_size': 0.016,  # 16mm bearing
                'range_min': limits[0],
                'range_max': limits[1]
            },
            material=material,
            mass=0.05,  # Approximate
            connection_points=[{'x': 0, 'y': 0, 'z': 0}],
            joint_type=joint_params.joint_type
        )
    
    def generate_assembly(
        self,
        morphology_config
    ) -> Tuple[List[CADComponent], List[AssemblyInstruction]]:
        """
        Generate complete assembly from morphology configuration.
        
        Returns:
            components: List of CAD components
            instructions: Assembly instructions
        """
        components = []
        instructions = []
        
        # Convert all links
        for i, link in enumerate(morphology_config.links):
            comp = self.convert_link_to_component(link, i)
            components.append(comp)
        
        # Convert all joints
        for i, joint in enumerate(morphology_config.joints):
            comp = self.convert_joint_to_component(joint, i)
            components.append(comp)
            
            # Create assembly instruction
            instr = AssemblyInstruction(
                component_a=f"link_{i}",
                component_b=f"link_{i+1}" if i + 1 < len(morphology_config.links) else f"link_{i}",
                connection_type='hinge' if joint.joint_type == 'revolute' else 'slider',
                torque_spec=1.5  # Nm
            )
            instructions.append(instr)
        
        return components, instructions
    
    def export_to_stl_metadata(self, components: List[CADComponent], filename: str):
        """
        Export component metadata for STL generation.
        
        Note: Actual STL generation would require a CAD kernel like OpenCASCADE.
        This exports the metadata needed for external CAD tools.
        """
        output_path = self.output_dir / f"{filename}_metadata.json"
        
        export_data = {
            'components': [asdict(c) for c in components],
            'units': 'meters',
            'coordinate_system': 'right_handed_z_up'
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported CAD metadata to: {output_path}")
        return output_path
    
    def export_to_openscad(
        self, 
        components: List[CADComponent], 
        filename: str
    ) -> Path:
        """
        Generate OpenSCAD script for 3D model visualization.
        
        OpenSCAD is used as it's scriptable and can be version-controlled.
        """
        output_path = self.output_dir / f"{filename}.scad"
        
        scad_lines = [
            "// Auto-generated by MorphoCompiler",
            f"// Components: {len(components)}",
            "",
            "$fn = 50;  // Smooth circles",
            ""
        ]
        
        # Generate modules for each component
        for i, comp in enumerate(components):
            scad_lines.append(f"// Component: {comp.name}")
            scad_lines.append(f"translate([0, 0, {i * 0.1}]) {{")
            
            if comp.geometry_type == 'cylinder':
                dims = comp.dimensions
                scad_lines.append(
                    f"  cylinder(h={dims['length']}, r={dims['radius']}, center=true);"
                )
            elif comp.geometry_type == 'box':
                dims = comp.dimensions
                scad_lines.append(
                    f"  cube([{dims.get('width', 0.1)}, {dims.get('depth', 0.1)}, {dims.get('height', 0.1)}], center=true);"
                )
            elif comp.geometry_type == 'sphere':
                dims = comp.dimensions
                scad_lines.append(
                    f"  sphere(r={dims.get('radius', 0.05)});"
                )
            
            scad_lines.append("}")
            scad_lines.append("")
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(scad_lines))
        
        print(f"Generated OpenSCAD file: {output_path}")
        return output_path
    
    def export_bom(self, components: List[CADComponent], filename: str = "bom") -> Path:
        """Generate Bill of Materials (BOM)."""
        output_path = self.output_dir / f"{filename}.csv"
        
        lines = ["name,type,material,mass_kg,length_m,radius_m"]
        
        for comp in components:
            dims = comp.dimensions
            lines.append(
                f"{comp.name},{comp.geometry_type},{comp.material},"
                f"{comp.mass:.4f},{dims.get('length', 0):.4f},{dims.get('radius', 0):.4f}"
            )
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Generated BOM: {output_path}")
        return output_path
    
    def estimate_cost(
        self, 
        components: List[CADComponent],
        labor_rate: float = 50.0  # $/hour
    ) -> Dict:
        """Estimate fabrication cost."""
        material_costs = {
            'pla': 20.0,  # $/kg
            'abs': 25.0,
            'nylon': 40.0,
            'carbon_fiber': 150.0,
            'aluminum': 30.0
        }
        
        total_material = 0.0
        total_print_time = 0.0  # hours
        
        for comp in components:
            mat_cost_per_kg = material_costs.get(comp.material, 50.0)
            total_material += comp.mass * mat_cost_per_kg
            
            # Rough print time estimate: 1 hour per 100g
            total_print_time += comp.mass * 10
        
        labor_cost = total_print_time * labor_rate
        total_cost = total_material + labor_cost
        
        return {
            'material_cost': total_material,
            'labor_cost': labor_cost,
            'print_time_hours': total_print_time,
            'total_cost': total_cost
        }


def generate_urdf(morphology_config, output_path: str):
    """
    Generate URDF (Unified Robot Description Format) for ROS compatibility.
    
    This enables simulation in Gazebo and deployment on real robots.
    """
    urdf_lines = ['<?xml version="1.0"?>', '<robot name="morpho_robot">']
    
    # Add links
    for i, link in enumerate(morphology_config.links):
        length = float(link.length[0])
        mass = float(link.mass[0])
        radius = link.radius
        
        urdf_lines.append(f'''
  <link name="link_{i}">
    <inertial>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{mass*length**2/12:.6f}" ixy="0" ixz="0" 
               iyy="{mass*length**2/12:.6f}" iyz="0" 
               izz="{mass*radius**2/2:.6f}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="{length:.4f}" radius="{radius:.4f}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{length:.4f}" radius="{radius:.4f}"/>
      </geometry>
    </collision>
  </link>''')
    
    # Add joints
    for i, joint in enumerate(morphology_config.joints):
        urdf_lines.append(f'''
  <joint name="joint_{i}" type="revolute">
    <parent link="link_{i}"/>
    <child link="link_{i+1}"/>
    <axis xyz="{joint.axis[0]} {joint.axis[1]} {joint.axis[2]}"/>
    <limit lower="{joint.lower_limit}" upper="{joint.upper_limit}" 
           effort="10.0" velocity="5.0"/>
  </joint>''')
    
    urdf_lines.append('</robot>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(urdf_lines))
    
    print(f"Generated URDF: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("MorphoCompiler Fabrication Module")
    print("=" * 60)
    
    from morphodsl import MorphoChain, planar_leg_morphology
    
    # Test 1: Convert morphology to CAD components
    print("\n1. Converting morphology to CAD components...")
    config = planar_leg_morphology()
    converter = MorphologyToCADConverter(output_dir="./test_cad")
    
    components, instructions = converter.generate_assembly(config)
    print(f"   Generated {len(components)} components")
    print(f"   Generated {len(instructions)} assembly instructions")
    
    # Test 2: Export to OpenSCAD
    print("\n2. Generating OpenSCAD visualization...")
    scad_path = converter.export_to_openscad(components, "test_robot")
    
    # Test 3: Generate BOM
    print("\n3. Generating Bill of Materials...")
    bom_path = converter.export_bom(components)
    
    # Test 4: Cost estimation
    print("\n4. Estimating fabrication cost...")
    cost = converter.estimate_cost(components)
    print(f"   Material cost: ${cost['material_cost']:.2f}")
    print(f"   Labor cost: ${cost['labor_cost']:.2f}")
    print(f"   Print time: {cost['print_time_hours']:.1f} hours")
    print(f"   Total cost: ${cost['total_cost']:.2f}")
    
    # Test 5: URDF generation
    print("\n5. Generating URDF for ROS...")
    urdf_path = converter.output_dir / "test_robot.urdf"
    generate_urdf(config, str(urdf_path))
    
    print("\n✓ Fabrication module tests passed!")
    print(f"\nOutput directory: {converter.output_dir.absolute()}")
