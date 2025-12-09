import numpy as np

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