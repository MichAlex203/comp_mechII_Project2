# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 13:19:04 2026

@author: Micha
"""

import numpy as np

class KirchhoffPlateElement:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.mat = material

    
    def w(self, x, y):
        """
        Defining w using Pascal's polyonym
        """
        w = [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**3*y, x*y**3]
        
        # theta_x = dw/dy
        thetax = [0, 0, 1, 0, x, 2*y, 0, x**2, 2*y*x, 3*y**2, x**3, 3*y**2*x]
        
        # theta_y = dw/dx
        thetay = [0, 1, 0, 2*x, y, 0, 3*x**2, 2*x*y, y**2, 0, 3*x**2*y, y**3]

        return w, thetax, thetay
    
    def A_inv(self, a=1.0, b=1.0):
        
        nodes = [
            (-a, -b), # Node 1
            ( a, -b), # Node 2
            ( a,  b), # Node 3
            (-a,  b)  # Node 4
        ]
        
        A_rows = []
        
        for x_node, y_node in nodes:
            w, thx, thy = self.w(x_node, y_node)
            
            A_rows.append(w)
            A_rows.append(thx)
            A_rows.append(thy)
            
        A = np.array(A_rows)
        
        A_inv = np.linalg.inv(A)
        return A_inv
    
       
    def beta(self, x, y):
        
        #kx = - d^2w/dx^2
        kx = [0, 0, 0, 2, 0, 0, 6*x, 2*y, 0, 0, 6*x*y, 0]
        #ky = - d^2w/dy^2
        ky = [0, 0, 0, 0, 0, 2, 0, 0, 2*x, 6*y, 0, 6*x*y]
        #kxy = -2 d^2w/dxdy
        kxy = [0, 0, 0, 0, 2, 0, 0, 4*x, 4*y, 0, 6*x**2, 6*y**2]
        
        b = np.array([kx, ky, kxy])
        
        return b
    
    
    def Te_inv(self, elem_nodes):
        
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]

        ax = x[1] - x[0]
        ay = y[1] - y[0]
        bx = x[3] - x[0]
        by = y[3] - y[0]
        
        J = 0.5*np.array([
            [ax, bx],
            [ay, by]
            ])
        
        J_inv = np.linalg.inv(J)
        
        Tn = np.zeros((3, 3))
        Tn[0, 0] = 1.0
        
        Tn[1, 1] = -J_inv[0, 0]
        Tn[1, 2] = -J_inv[0, 1]
        Tn[2, 1] = -J_inv[1, 0]
        Tn[2, 2] = -J_inv[1, 1]
        
        Te = np.zeros((12, 12))
        for i in range(4):
            Te[3*i : 3*i+3, 3*i : 3*i+3] = Tn
                
        Te_inv = np.linalg.inv(Te)
        
        return Te_inv
        
        
    def Tb(self, elem_nodes):
        
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]

        ax = x[1] - x[0]
        ay = y[1] - y[0]
        bx = x[3] - x[0]
        by = y[3] - y[0]
        
        D = ax*by - ay*bx
        
        Tb = (4.0/D**2)*np.array([
            [  by**2 ,   ay**2 ,   -ay*by],
            [  bx**2 ,   ax**2 ,   -ax*bx],
            [-2*bx*by, -2*ax*ay, (ax*by+ay*bx)]
            ])
    
        return Tb
    
    
    def ke(self, elem_nodes):
        
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]
        
        ax = x[1] - x[0]
        ay = y[1] - y[0]
        bx = x[3] - x[0]
        by = y[3] - y[0]
        
        D = ax*by - ay*bx
        detJ = D/4.0
        
        Ek = self.mat.Ek
        
        Te_inv = self.Te_inv(elem_nodes)
        Tb = self.Tb(elem_nodes)
        
        # Ολοκλήρωση Gauss
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]
        ke = np.zeros((12,12))
        
        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i]*w[j]
                
                A1 = self.A_inv()

                beta = self.beta(xi,eta)
                
                BT = beta @ A1
                Bx = Tb @ BT @ Te_inv
                
                ke += Bx.T @ Ek @ Bx * abs(detJ) * weight
                
        return ke
    
    
    def q_loading(self, elem_nodes, q):
        
        f = np.zeros(12)
        
        x = elem_nodes[:, 0]
        y = elem_nodes[:, 1]

        ax = x[1] - x[0]
        ay = y[1] - y[0]
        bx = x[3] - x[0]
        by = y[3] - y[0]

        detJ = (ax * by - ay * bx) / 4.0

        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        w  = [1, 1]
        
        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i] * w[j]

                A1 = self.A_inv()
                w_vec, _, _ = self.w(xi,eta)

                N = np.array(w_vec) @ A1
                
                f += N.T * q *abs(detJ)* weight
        
        return f
    
 
    def calculate_stress(self, elem_nodes, u_elem, z_coord, xi=1, eta=1):
        stresses = np.zeros(3)
        
        t = self.mat.t
        E = self.mat.Ek * (12.0 / t**3)
        
        Te_inv = self.Te_inv(elem_nodes)
        Tb = self.Tb(elem_nodes)
    
        A1 = self.A_inv()
        beta = self.beta(xi,eta)
               
        BT = beta @ A1
        Bx = Tb @ BT @ Te_inv
        S_matrix = E @ Bx * z_coord
        
        stresses = S_matrix @ u_elem
        
        return stresses.flatten()


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