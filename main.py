from pre_processor import Mesh, Material, BoundaryConditions
from solver_parallel import KirchhoffPlateElement, Assembler, Solver
from post_processor import PostProcessor
from interactive_plots import plot_mesh_quad_interactive, plot_displacement_interactive
from interactive_plots import plot_deformed_shape, plot_displacement_3d, plot_stress_map
from meshGenerator import generate_parallelogram_mesh
import numpy as np

if __name__ == "__main__":
    
    """
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
    """
    a = 1.0
    b = 1.0
    theta = 30
    rad_theta = np.deg2rad(theta)
    
    N_mesh = 32
    nodes, elements = generate_parallelogram_mesh(a, b, theta, N_mesh, N_mesh)

    mesh = Mesh(nodes, elements)
    mat = Material(E=210e9, nu=0.3, t=0.01)
    
    # Checking mesh
    plot_mesh_quad_interactive(nodes, elements, show=True, filename='Figures - Results/Interactive_mesh.html')

    element = KirchhoffPlateElement(mesh, mat)
    asm = Assembler(mesh, element)
    K = asm.assemble_stiffness()

    F = np.zeros(mesh.ndof)
    bc = BoundaryConditions()
    
    # uniform load
    F = bc.apply_consistent_load(F, mesh, element, q=-700)

    """
    bc.fixed.append((0,0))
    bc.fixed.append((0,1))
    bc.fixed.append((0,2))
    """
    
    # Boundary conditions
    tol = 1e-6
    for i, node in enumerate(nodes):
        x, y = node
        # Έλεγχος αν ο κόμβος είναι στα όρια του παραλληλογράμμου
        # Κάτω Πλευρά (y=0)
        bottom = (abs(y) < tol)
    
        # Πάνω Πλευρά (y=height)
        top = (abs(y - b * np.sin(rad_theta)) < tol)
        
        # Υπολογισμός του x που αντιστοιχεί σε αυτό το y στην αριστερή πλευρά
        x_left = y / np.tan(rad_theta) 
        left = (abs(x - x_left) < tol)
        
        # Δεξιά πλευρά (μετατόπιση +1 στο x)
        right = (abs(x - (x_left + 1.0)) < tol) 
        
        if bottom or top or left or right:
            # Δέσμευση μόνο του w (dof 0)
            # Προσοχή: Ελέγξτε αν το dof 0 είναι το w ή αν είναι τα u,v,w. 
            # Στις πλάκες Kirchhoff συνήθως είναι (w, theta_x, theta_y).
            bc.fixed.append((i, 0))
        
    
    # Solve Problem
    K, F = bc.apply_bc(K, F)

    solver = Solver()
    U = solver.solve(K, F)

    post = PostProcessor(mesh)
    d = post.export_displacements(U)
    print("Displacements:\n", d)
    
    # Checking displacements
    plot_displacement_interactive(nodes, elements, U, scale=1, filename='Figures - Results/disp.html')
    plot_deformed_shape(nodes, elements, U, scale=1, filename='Figures - Results/disp.png')
    plot_displacement_3d(np.array(nodes), U, filename='Figures - Results/disp_3d.png')

    # Printing displacements
    post = PostProcessor(mesh)
    d = post.export_displacements(U)

    post.export_displacements_csv(np.array(nodes), d, filename='Figures - Results/displacements.csv')
    
    # Εξαγωγή Τάσεων (CSV + PNG)
    post.compute_and_export_stresses(
        U, 
        element, 
        filename_csv='Figures - Results/stresses.csv', 
        filename_png='Figures - Results/stresses_von_mises.png'
    )
    
    post.calculate_center_stress(U, element)