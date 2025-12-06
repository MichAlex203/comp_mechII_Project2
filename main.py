from pre_processor import Mesh, Material, BoundaryConditions
from solver import KirchhoffPlateElement, Assembler, Solver
from post_processor import PostProcessor
from interactive_plots import plot_mesh_quad_interactive, plot_displacement_interactive, plot_deformed_shape
import numpy as np

if __name__ == "__main__":
    
    theta = np.deg2rad(30)
    Lx = 1.0
    Ly = 1.0

    nodes = [
    (0.0, 0.0),
    (Lx, 0.0),
    (Lx + Ly*np.cos(theta), Ly*np.sin(theta)),
    (Ly*np.cos(theta), Ly*np.sin(theta))
    ]

    elements = [(0,1,2,3)]

    mesh = Mesh(nodes, elements)
    mat = Material(E=210e9, nu=0.3, t=0.01)
    
    # Checking mesh
    plot_mesh_quad_interactive(nodes, elements, show=True, filename='Interactive_mesh.html')

    element = KirchhoffPlateElement(mesh, mat)
    asm = Assembler(mesh, element)
    K = asm.assemble_stiffness()

    F = np.zeros(mesh.ndof)

    bc = BoundaryConditions()
    # uniform load
    bc.add_uniform_pressure(q=-700, mesh=mesh)

    bc.fixed.append((0,0))
    bc.fixed.append((0,1))
    bc.fixed.append((0,2))
    K, F = bc.apply(K, F)

    solver = Solver()
    U = solver.solve(K, F)

    post = PostProcessor(mesh)
    d = post.export_displacements(U)
    print("Displacements:\n", d)
    
    # Checking displacements
    # plot_displacement_interactive(nodes, elements, U, scale=1e4, filename='disp.html')
    plot_deformed_shape(nodes, elements, U, scale=1, filename='disp.png')
