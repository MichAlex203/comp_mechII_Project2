import numpy as np

class KirchhoffPlateElement:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.mat = material

    def ke(self, elem_nodes):
        " Define material parameters "
      
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]

        a = 0.5*(max(x) - min(x))
        b = 0.5*(max(y) - min(y))
        
        " Define Ek "
        # Dk = self.mat.D
        Ek = self.mat.Ek
        
        " Define A^-1 "
        scalar = 1/(8*a**3*b**3)
        A_inv = scalar * np.array([
        # Γραμμή 1 
        [2*a**3*b**3, a**3*b**4, a**4*b**3, 2*a**3*b**3, a**3*b**4, -a**4*b**3, 2*a**3*b**3, -a**3*b**4, -a**4*b**3, 2*a**3*b**3, -a**3*b**4, a**4*b**3],
        # Γραμμή 2
        [-3*a**2*b**3, -a**2*b**4, -a**3*b**3, 3*a**2*b**3, a**2*b**4, -a**3*b**3, 3*a**2*b**3, -a**2*b**4, -a**3*b**3, -3*a**2*b**3, a**2*b**4, -a**3*b**3],
        # Γραμμή 3
        [-3*a**3*b**2, -a**3*b**3, -a**4*b**2, -3*a**3*b**2, -a**3*b**3, a**4*b**2, 3*a**3*b**2, -a**3*b**3, -a**4*b**2, 3*a**3*b**2, -a**3*b**3, a**4*b**2],
        # Γραμμή 4
        [0, 0, -a**2*b**3, 0, 0, a**2*b**3, 0, 0, a**2*b**3, 0, 0, -a**2*b**3],
        # Γραμμή 5
        [4*a**2*b**2, a**2*b**3, a**3*b**2, -4*a**2*b**2, -a**2*b**3, a**3*b**2, 4*a**2*b**2, -a**2*b**3, -a**3*b**2, -4*a**2*b**2, a**2*b**3, -a**3*b**2],
        # Γραμμή 6
        [0, -a**3*b**2, 0, 0, -a**3*b**2, 0, 0, a**3*b**2, 0, 0, a**3*b**2, 0],
        # Γραμμή 7
        [b**3, 0, a*b**3, -b**3, 0, a*b**3, -b**3, 0, a*b**3, b**3, 0, a*b**3],
        # Γραμμή 8
        [0, 0, a**2*b**2, 0, 0, -a**2*b**2, 0, 0, a**2*b**2, 0, 0, -a**2*b**2],
        # Γραμμή 9
        [0, a**2*b**2, 0, 0, -a**2*b**2, 0, 0, a**2*b**2, 0, 0, -a**2*b**2, 0],
        # Γραμμή 10
        [a**3, a**3*b, 0, a**3, a**3*b, 0, -a**3, a**3*b, 0, -a**3, a**3*b, 0],
        # Γραμμή 11
        [-b**2, 0, -a*b**2, b**2, 0, -a*b**2, -b**2, 0, a*b**2, b**2, 0, a*b**2],
        # Γραμμή 12
        [-a**2, -a**2*b, 0, a**2, a**2*b, 0, -a**2, a**2*b, 0, a**2, -a**2*b, 0]
        ])
        
        " Gauss integration for B "
        # 2×2 Gauss points
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]

        ke = np.zeros((12,12))

        # loop over Gauss points
        for i in range(2):
            for j in range(2):
                x_new = gp[i]
                y_new = gp[j]
                weight = w[i]*w[j]

                # β matrix (3×12)
                beta = np.array([
                    [0,0,0, 2,0,0, 6*x_new,2*y_new,0, 0,6*x_new*y_new,0],
                    [0,0,0, 0,0,2, 0,0,2*x_new, 6*y_new,0,6*x_new*y_new],
                    [0,0,0, 0,2,0, 0,4*x_new,4*y_new, 0,6*x_new**2,6*y_new**2]
                ])

                # B matrix
                B = beta @ A_inv

                # integration weight: |J| = a*b
                detJ = a*b

                # add contribution
                ke += B.T @ Ek @ B * detJ * weight

        return ke
        

class Assembler:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element

    def assemble_stiffness(self):
        K = np.zeros((self.mesh.ndof, self.mesh.ndof))
        for conn in self.mesh.elements:
            elem_coords = self.mesh.nodes[conn]
            ke = self.element.ke(elem_coords)
            edofs = []
            for nid in conn:
                base = nid * self.mesh.ndof_per_node
                edofs.extend([base, base+1, base+2])
            for i in range(12):
                for j in range(12):
                    K[edofs[i], edofs[j]] += ke[i, j]
        return K

class Solver:
    def solve(self, K, F):
        return np.linalg.solve(K, F)
