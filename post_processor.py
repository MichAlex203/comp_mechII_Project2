import numpy as np
from interactive_plots import plot_stress_map

class PostProcessor:
    def __init__(self, mesh):
        self.mesh = mesh

    def export_displacements(self, U):
        disp = U.reshape((-1, 3))
        return disp

    def export_displacements_csv(self, nodes, disp, filename='displacements.csv'):
        """
        Export nodal displacements to CSV.
        Columns: x, y, w, theta_x, theta_y
        """
        data = np.column_stack([nodes[:, 0], nodes[:, 1], disp[:, 0], disp[:, 1], disp[:, 2]])
        np.savetxt(filename, data, delimiter=',', header='x,y,w,theta_x,theta_y', comments='')
        return filename
    
    def compute_and_export_stresses(self, U, element_solver, filename_csv='stresses.csv', filename_png='stresses.png'):
        """
        Υπολογίζει τις τάσεις σε 5 σημεία ανά στοιχείο (Κέντρο + 4 Κόμβοι).
        Εξάγει τα αποτελέσματα σε CSV και δημιουργεί εικόνα PNG.
        """
        print("Calculating stresses for all elements...")
        
        # Ζητάμε τάσεις στην άνω επιφάνεια (εκεί που συνήθως είναι μέγιστες)
        z_top = element_solver.mat.t / 2.0
        
        # Λίστα για αποθήκευση δεδομένων (για το CSV)
        # Format: [ElemID, LocalPointID, x, y, sx, sy, txy, VonMises]
        all_data = []

        # Ορισμός των 5 σημείων σε φυσικές συντεταγμένες (xi, eta)
        # Local ID: 0=Center, 1=Node1(-1,-1), 2=Node2(1,-1), 3=Node3(1,1), 4=Node4(-1,1)
        # Προσοχή: Η σειρά των κόμβων στοιχείου είναι συνήθως CCW: (-1,-1), (1,-1), (1,1), (-1,1)
        points_of_interest = [
            (0, 0.0, 0.0),    # Center
            (1, -1.0, -1.0),  # Node 1
            (2, 1.0, -1.0),   # Node 2
            (3, 1.0, 1.0),    # Node 3
            (4, -1.0, 1.0)    # Node 4
        ]

        for elem_idx, conn in enumerate(self.mesh.elements):
            elem_nodes = self.mesh.nodes[conn]
            
            # Ανάκτηση μετατοπίσεων στοιχείου
            edofs = []
            for nid in conn:
                base = nid * self.mesh.ndof_per_node
                edofs.extend([base, base+1, base+2])
            u_elem = U[edofs]

            # Υπολογισμός για κάθε ένα από τα 5 σημεία
            for pid, xi, eta in points_of_interest:
                # 1. Υπολογισμός Τάσεων
                sigma = element_solver.calculate_stresses(elem_nodes, u_elem, z_coord=z_top, xi=xi, eta=eta)
                sx, sy, txy = sigma[0], sigma[1], sigma[2]
                
                # 2. Von Mises Stress
                vm = np.sqrt(sx**2 + sy**2 - sx*sy + 3*txy**2)

                # 3. Εύρεση φυσικών συντεταγμένων (x, y) για το export
                if pid == 0:
                    # Κέντρο: Μέσος όρος των συντεταγμένων των κόμβων
                    x_phys = np.mean(elem_nodes[:, 0])
                    y_phys = np.mean(elem_nodes[:, 1])
                else:
                    # Κόμβοι: Παίρνουμε απευθείας τις συντεταγμένες του αντίστοιχου κόμβου
                    # Το pid αντιστοιχεί στο index του κόμβου στο connectivity (1->0, 2->1, etc.)
                    node_idx_in_elem = pid - 1
                    x_phys = elem_nodes[node_idx_in_elem, 0]
                    y_phys = elem_nodes[node_idx_in_elem, 1]

                # Αποθήκευση
                all_data.append([elem_idx, pid, x_phys, y_phys, sx, sy, txy, vm])

        # --- Εξαγωγή σε CSV ---
        data_arr = np.array(all_data)
        header = 'ElemID,PointID,x,y,sigma_x,sigma_y,tau_xy,VonMises'
        np.savetxt(filename_csv, data_arr, delimiter=',', header=header, comments='')
        print(f"Stresses exported to {filename_csv}")

        # --- Δημιουργία PNG (Heatmap) ---
        plot_stress_map(data_arr, filename=filename_png, show=False)
        
    def calculate_center_stress(self, U, element_solver):
        """
        Υπολογίζει την Κύρια Τάση (Max Principal Stress) στην κάτω επιφάνεια
        στο γεωμετρικό κέντρο της πλάκας.
        """
        print("\n--- Calculation: Center Point, Bottom Surface ---")
        
        nodes = self.mesh.nodes
        elements = self.mesh.elements

        # 1. Εύρεση του γεωμετρικού κέντρου (τομή διαγωνίων)
        center_x = np.mean(nodes[:, 0])
        center_y = np.mean(nodes[:, 1])
        
        # 2. Εύρεση του στοιχείου που βρίσκεται πιο κοντά στο κέντρο
        closest_elem_idx = -1
        min_dist = 1e9
        
        for i, conn in enumerate(elements):
            el_nodes = nodes[conn]
            el_cx = np.mean(el_nodes[:, 0])
            el_cy = np.mean(el_nodes[:, 1])
            
            dist = np.sqrt((el_cx - center_x)**2 + (el_cy - center_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_elem_idx = i

        # 3. Υπολογισμός τάσεων στο κέντρο αυτού του στοιχείου
        # Ζητείται στην κατώτερη επιφάνεια => z = -t/2
        thickness = element_solver.mat.t
        z_bottom = -thickness / 2.0
        
        conn = elements[closest_elem_idx]
        el_nodes = nodes[conn]
        
        # Ανάκτηση μετατοπίσεων
        edofs = []
        for nid in conn:
            base = nid * self.mesh.ndof_per_node
            edofs.extend([base, base+1, base+2])
        u_elem = U[edofs]

        # Υπολογισμός [sx, sy, txy]
        sigma = element_solver.calculate_stresses(el_nodes, u_elem, z_coord=z_bottom, xi=0.0, eta=0.0)
        sx, sy, txy = sigma[0], sigma[1], sigma[2]

        # 4. Υπολογισμός Μέγιστης Κύριας Τάσης (Principal Stress σ1)
        # Τύπος: center + radius (Κύκλος Mohr)
        center_stress = (sx + sy) / 2.0
        radius_stress = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
        
        sigma1 = center_stress + radius_stress

        # Μετατροπή σε MPa
        sigma1_mpa = sigma1 / 1e6

        print(f"Geometric Center:      x={center_x:.3f}, y={center_y:.3f}")
        print(f"Checked Element ID:    {closest_elem_idx}")
        print(f"Stresses (z={z_bottom:.4f}):")
        print(f"  Sigma_x: {sx:.2f} Pa")
        print(f"  Sigma_y: {sy:.2f} Pa")
        print(f"  Tau_xy:  {txy:.2f} Pa")
        print(f"Max Principal (σ1):    {sigma1_mpa:.4f} MPa")
            
        return sigma1_mpa