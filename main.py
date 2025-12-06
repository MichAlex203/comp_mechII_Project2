from pre_processor import Mesh, Material, BoundaryConditions
from solver import KirchhoffPlateElement, Assembler, Solver
from post_processor import PostProcessor
import numpy as np

if __name__ == "__main__":
    nodes = [(0,0), (1,0), (1,1), (0,1)]
    elements = [(0,1,2,3)]

    mesh = Mesh(nodes, elements)
    mat = Material(E=210e9, nu=0.3, t=0.01)

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
