import numpy as np

class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements, dtype=int)
        self.ndof_per_node = 3
        self.ndof = self.nodes.shape[0] * self.ndof_per_node

class Material:
    def __init__(self, E, nu, t):
        self.E = E
        self.nu = nu
        self.t = t
        self.D = E * t**3 / (12 * (1 - nu**2))
        self.Ek = self.D * np.array([[1,   nu,   0],
                                     [nu,   1,   0],
                                     [0,   0, 0.5*(1 - nu)]])

class BoundaryConditions:
    def __init__(self):
        self.fixed = []
        self.loads = {}

    def apply_consistent_load(self, F, mesh, element_solver, q):
        """
        Εφαρμόζει το ομοιόμορφο φορτίο q χρησιμοποιώντας το Consistent Load Vector.
        """
        ndof_per_node = mesh.ndof_per_node
        
        for conn in mesh.elements:
            elem_nodes = mesh.nodes[conn]
            
            # Καλούμε τη νέα συνάρτηση του στοιχείου
            fe = element_solver.q_loading(elem_nodes, q)
            
            # Assembly στο global F vector
            edofs = []
            for nid in conn:
                base = nid * ndof_per_node
                edofs.extend([base, base+1, base+2])
            
            for i in range(12):
                F[edofs[i]] += fe[i]
        
        return F

    def apply_bc(self, K, F, ndof_per_node=3):
        # Εφαρμογή πακτώσεων (Fixed DOFs)
        for (nid, d) in self.fixed:
            gdof = nid*ndof_per_node + d
            K[gdof, :] = 0.0
            K[:, gdof] = 0.0
            K[gdof, gdof] = 1.0
            F[gdof] = 0.0
        return K, F
