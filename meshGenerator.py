# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 03:01:01 2025

@author: Micha
"""

import numpy as np

def generate_parallelogram_mesh(Lx, Ly, theta_deg, nx, ny):
    """
    Δημιουργεί πλέγμα τετραγωνικών στοιχείων για παραλληλόγραμμη πλάκα.

    Παράμετροι:
    -----------
    Lx, Ly : float
        Μήκη πλευρών.
    theta_deg : float
        Γωνία στο κάτω αριστερό κόμβο (σε μοίρες).
    nx, ny : int
        Αριθμός στοιχείων κατά x και y.

    Επιστρέφει:
    -----------
    nodes : (N,2) array
        Συντεταγμένες κόμβων.
    elements : list of tuples
        Λίστα στοιχείων με 4 κόμβους το καθένα.
    """
    theta = np.deg2rad(theta_deg)

    # Διανύσματα πλευρών
    vec_x = np.array([Lx, 0])
    vec_y = np.array([Ly*np.cos(theta), Ly*np.sin(theta)])

    # Δημιουργία κόμβων
    nodes = []
    for j in range(ny+1):
        for i in range(nx+1):
            point = i/nx*vec_x + j/ny*vec_y
            nodes.append(point)
    nodes = np.array(nodes)

    # Δημιουργία στοιχείων
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1) + i
            n1 = n0 + 1
            n3 = n0 + (nx+1)
            n2 = n3 + 1
            elements.append((n0, n1, n2, n3))

    return nodes, elements
