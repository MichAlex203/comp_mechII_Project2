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
    
    def dof_transformation_matrix(self, ax, ay, bx, by):
        """
        Κατασκευάζει τον πίνακα μετασχηματισμού των βαθμών ελευθερίας (T_dof).
        Μετατρέπει τα DOFs από το Global (w, w_x, w_y) στο Local (w, w_xi, w_eta).
        d_local = T_dof * d_global
        
        Βάσει του Chain Rule:
        w_xi  = w_x * (dx/dxi) + w_y * (dy/dxi)
        w_eta = w_x * (dx/deta) + w_y * (dy/deta)
        """
        # Οι μερικές παράγωγοι της γεωμετρίας (Jacobian terms)
        # x = ... + xi * (ax/2) + ... => dx/dxi = ax/2
        dxdxi = ax / 2.0
        dydxi = ay / 2.0
        dxdeta = bx / 2.0
        dydeta = by / 2.0

        # Μπλοκ μετασχηματισμού για έναν κόμβο (3x3)
        # [ w   ]       [ 1     0       0     ] [ w   ]
        # [ w_xi]   =   [ 0   dx/dxi  dy/dxi  ] [ w_x ]
        # [ w_eta]      [ 0   dx/deta dy/deta ] [ w_y ]
        
        T_node = np.array([
            [1.0, 0.0,    0.0],
            [0.0, dxdxi,  dydxi],
            [0.0, dxdeta, dydeta]
        ])

        # Κατασκευή του πλήρους πίνακα 12x12 (Block Diagonal)
        T_dof = np.zeros((12, 12))
        for i in range(4):
            T_dof[3*i : 3*i+3, 3*i : 3*i+3] = T_node
            
        return T_dof
    
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
        
        # 1. Υπολογισμός T_dof
        T_dof = self.dof_transformation_matrix(ax, ay, bx, by)

        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]
        
        # Αυτό είναι το f στο τοπικό σύστημα (local DOFs)
        f_local = np.zeros(12)

        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i] * w[j]

                p_vec = np.array([
                    1, xi, eta, 
                    xi**2, xi*eta, eta**2, 
                    xi**3, xi**2*eta, xi*eta**2, eta**3, 
                    xi**3*eta, xi*eta**3
                ])

                # N shape functions για local dofs
                N = p_vec @ self.A_inv

                f_local += N * q * abs(detJ) * weight
        
        # 2. Μετατροπή στο Global σύστημα
        # f_global = T_dof.T * f_local
        f_global = T_dof.T @ f_local

        return f_global

    
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
        
        # 1. Υπολογισμός T_dof 
        T_dof = self.dof_transformation_matrix(ax, ay, bx, by)

        # Ολοκλήρωση Gauss
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]
        ke_local = np.zeros((12,12))

        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i]*w[j]
                
                beta = np.array([
                    [0,0,0, 2,0,0, 6*xi, 2*eta, 0, 0, 6*xi*eta, 0],          
                    [0,0,0, 0,0,2, 0, 0, 2*xi, 6*eta, 0, 6*xi*eta],          
                    [0,0,0, 0,2,0, 0, 4*xi, 4*eta, 0, 6*xi**2, 6*eta**2]     
                ])

                B_nat = beta @ self.A_inv
                B_cart = Te @ B_nat

                ke_local += B_cart.T @ Ek @ B_cart * abs(detJ) * weight

        # 2. Μετατροπή στο Global σύστημα (w, w_x, w_y)
        # K_global = T_dof.T * K_local * T_dof
        ke_global = T_dof.T @ ke_local @ T_dof

        return ke_global
    
    def calculate_stresses(self, elem_nodes, u_elem_global, z_coord, xi=0, eta=0):
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
        
        # Πρέπει να μετατρέψουμε και εδώ τα DOFs εισόδου
        T_dof = self.dof_transformation_matrix(ax, ay, bx, by)
        
        # Μετατροπή των global displacements σε local (για να πολλαπλασιαστούν με το B_nat/B_cart)
        u_elem_local = T_dof @ u_elem_global

        # Beta matrix στο σημείο (xi, eta)
        beta = np.array([
            [0,0,0, 2,0,0, 6*xi, 2*eta, 0, 0, 6*xi*eta, 0],
            [0,0,0, 0,0,2, 0, 0, 2*xi, 6*eta, 0, 6*xi*eta],
            [0,0,0, 0,2,0, 0, 4*xi, 4*eta, 0, 6*xi**2, 6*eta**2]
        ])

        # Υπολογισμός Καμπυλοτήτων {k} = [Te] * [B_nat] * {d}
        B_nat = beta @ self.A_inv
        B_cart = Te @ B_nat
        
        curvatures = B_cart @ u_elem_local  # {k_x, k_y, k_xy}^T

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
