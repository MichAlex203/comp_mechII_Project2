import numpy as np

class KirchhoffPlateElement:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.mat = material
        self.A_inv = self.A_inv()
        
    def A_inv(self):

        a = 1.0
        b = 1.0
        scalar = 1/(8*a**3*b**3)
        return scalar * np.array([
            [2*a**3*b**3, a**3*b**4, a**4*b**3, 2*a**3*b**3, a**3*b**4, -a**4*b**3, 2*a**3*b**3, -a**3*b**4, -a**4*b**3, 2*a**3*b**3, -a**3*b**4, a**4*b**3],
            [-3*a**2*b**3, -a**2*b**4, -a**3*b**3, 3*a**2*b**3, a**2*b**4, -a**3*b**3, 3*a**2*b**3, -a**2*b**4, -a**3*b**3, -3*a**2*b**3, a**2*b**4, -a**3*b**3],
            [-3*a**3*b**2, -a**3*b**3, -a**4*b**2, -3*a**3*b**2, -a**3*b**3, a**4*b**2, 3*a**3*b**2, -a**3*b**3, -a**4*b**2, 3*a**3*b**2, -a**3*b**3, a**4*b**2],
            [0, 0, -a**2*b**3, 0, 0, a**2*b**3, 0, 0, a**2*b**3, 0, 0, -a**2*b**3],
            [4*a**2*b**2, a**2*b**3, a**3*b**2, -4*a**2*b**2, -a**2*b**3, a**3*b**2, 4*a**2*b**2, -a**2*b**3, -a**3*b**2, -4*a**2*b**2, a**2*b**3, -a**3*b**2],
            [0, -a**3*b**2, 0, 0, -a**3*b**2, 0, 0, a**3*b**2, 0, 0, a**3*b**2, 0],
            [b**3, 0, a*b**3, -b**3, 0, a*b**3, -b**3, 0, a*b**3, b**3, 0, a*b**3],
            [0, 0, a**2*b**2, 0, 0, -a**2*b**2, 0, 0, a**2*b**2, 0, 0, -a**2*b**2],
            [0, a**2*b**2, 0, 0, -a**2*b**2, 0, 0, a**2*b**2, 0, 0, -a**2*b**2, 0],
            [a**3, a**3*b, 0, a**3, a**3*b, 0, -a**3, a**3*b, 0, -a**3, a**3*b, 0],
            [-b**2, 0, -a*b**2, b**2, 0, -a*b**2, -b**2, 0, a*b**2, b**2, 0, a*b**2],
            [-a**2, -a**2*b, 0, a**2, a**2*b, 0, -a**2, a**2*b, 0, a**2, -a**2*b, 0]
        ])
    
    def Te(self, ax, ay, bx, by, detJ):
        """
        Κατασκευή του πίνακα μετασχηματισμού Te[cite: 28].
        Συνδέει τις καμπυλότητες (x,y) με τις (xi, eta).
        k_x = Te * k_xi
        """
      
        D = 4.0 * detJ        
        coef = 4.0 / (D**2) 

        Te = np.zeros((3, 3))
        
        # Σειρά 1: d2w/dx2
        Te[0, 0] = by**2 
        Te[0, 1] = ay**2
        Te[0, 2] = -ay * by

        # Σειρά 2: d2w/dy2
        Te[1, 0] = bx**2
        Te[1, 1] = ax**2
        Te[1, 2] = -ax * bx

        # Σειρά 3: 2*d2w/dxdy
        Te[2, 0] = -2 * bx * by
        Te[2, 1] = -2 * ax * ay
        Te[2, 2] = (ax * by + ay * bx)

        return Te * coef
    
    def q_loading(self, elem_nodes, q):
        """
        Υπολογίζει το διάνυσμα φορτίου σύμφωνα με το PDF.
        f = integral( N.T * q * detJ ) dxi deta
        """
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]
        
        ax = x[1] - x[0]
        ay = y[1] - y[0]
        bx = x[3] - x[0]
        by = y[3] - y[0]

        # Jacobian
        detJ = (ax * by - ay * bx) / 4.0
        
        # Ολοκλήρωση Gauss
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]
        f_elem = np.zeros(12)

        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i] * w[j]

                # Διάνυσμα πολυωνύμων P(xi, eta)
                p_vec = np.array([
                    1, xi, eta, 
                    xi**2, xi*eta, eta**2, 
                    xi**3, xi**2*eta, xi*eta**2, eta**3, 
                    xi**3*eta, xi*eta**3
                ])

                # Συναρτήσεις σχήματος: N = P * A_inv
                N = p_vec @ self.A_inv

                # Προσθήκη στο ολοκλήρωμα: N^T * q * detJ * weight
                f_elem += N * q * abs(detJ) * weight

        return f_elem
    
    
    def ke(self, elem_nodes):
        
        " Define material parameters "
      
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]
        
        # Μητρώο υλικού
        Ek = self.mat.Ek
        
        " 1. Υπολογισμός διανυσμάτων πλευρών a και b "
        # vec_a: Πλευρά κατά ξ (Node 1 -> Node 2)
        ax = x[1] - x[0]
        ay = y[1] - y[0]
        
        # vec_b: Πλευρά κατά η (Node 1 -> Node 4)
        bx = x[3] - x[0]
        by = y[3] - y[0]

        " 2. Ιακωβιανή Ορίζουσα (detJ) για την ολοκλήρωση "
        detJ = (ax * by - ay * bx) / 4.0

        " 3. Πίνακας Μετασχηματισμού Te "
        Te = self.Te(ax, ay, bx, by, detJ)

        # Ολοκλήρωση Gauss
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]
        ke = np.zeros((12,12))

        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i]*w[j]

                # Beta matrix (παράγωγοι ως προς xi, eta)
                
                beta = np.array([
                    [0,0,0, 2,0,0, 6*xi, 2*eta, 0, 0, 6*xi*eta, 0],          # d2/dxi2
                    [0,0,0, 0,0,2, 0, 0, 2*xi, 6*eta, 0, 6*xi*eta],          # d2/deta2
                    [0,0,0, 0,2,0, 0, 4*xi, 4*eta, 0, 6*xi**2, 6*eta**2]     # 2*d2/dxi*deta
                ])

                # B_natural: Καμπυλότητες στο φυσικό σύστημα (xi, eta)
                B_nat = beta @ self.A_inv

                # B_cartesian: Καμπυλότητες στο καρτεσιανό σύστημα (x, y)
                # k_x = Te * k_xi
                B_cart = Te @ B_nat

                ke += B_cart.T @ Ek @ B_cart * abs(detJ) * weight

        return ke
    
    def calculate_stresses(self, elem_nodes, u_elem, z_coord, xi=0, eta=0):
        """
        Υπολογίζει τις τάσεις [sigma_x, sigma_y, tau_xy] σε συγκεκριμένο σημείο (xi, eta, z).
        [cite_start]sigma = [E] * z * [B] * d 
        """
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]
        
        ax = x[1] - x[0]
        ay = y[1] - y[0]
        bx = x[3] - x[0]
        by = y[3] - y[0]

        # Ιακωβιανή
        detJ = (ax * by - ay * bx) / 4.0

        # Πίνακας Μετασχηματισμού Te
        Te = self.Te(ax, ay, bx, by, detJ)

        # Beta matrix στο σημείο (xi, eta)
        beta = np.array([
            [0,0,0, 2,0,0, 6*xi, 2*eta, 0, 0, 6*xi*eta, 0],
            [0,0,0, 0,0,2, 0, 0, 2*xi, 6*eta, 0, 6*xi*eta],
            [0,0,0, 0,2,0, 0, 4*xi, 4*eta, 0, 6*xi**2, 6*eta**2]
        ])

        # Υπολογισμός Καμπυλοτήτων {k} = [Te] * [B_nat] * {d}
        B_nat = beta @ self.A_inv
        B_cart = Te @ B_nat
        
        curvatures = B_cart @ u_elem  # {k_x, k_y, k_xy}^T

        # Υπολογισμός Μητρώου Ελαστικότητας Επιπέδου (Plane Stress Matrix)
        t = self.mat.t
        E_plane = self.mat.Ek * (12.0 / t**3)

        # Υπολογισμός Τάσεων: {sigma} = [E_plane] * z * {k}
        stresses = E_plane @ curvatures * z_coord
        
        return stresses
        

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
