import numpy as np

class PostProcessor:
    def __init__(self, mesh):
        self.mesh = mesh

    def export_displacements(self, U):
        disp = U.reshape((-1, 3))
        return disp
