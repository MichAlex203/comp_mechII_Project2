# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 23:29:51 2025

@author: Micha
"""

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

def plot_mesh_quad_interactive(nodes, elems, show=True, filename=None):
    """
    Interactive quadrilateral mesh plot with node indices.

    Parameters
    ----------
    nodes : ndarray (N,2)
        Node coordinates (x,y)
    elems : list of tuples/lists
        Each element is (n0, n1, n2, n3)
    """
    nodes = np.asarray(nodes)
    x = nodes[:, 0]
    y = nodes[:, 1]

    fig = go.Figure()

    # Draw each quadrilateral
    for quad in elems:
        quad_x = x[list(quad)].tolist() + [x[quad[0]]]  # close loop
        quad_y = y[list(quad)].tolist() + [y[quad[0]]]
        fig.add_trace(go.Scatter(
            x=quad_x,
            y=quad_y,
            mode='lines',
            line=dict(color='black'),
            showlegend=False
        ))

    # Draw nodes + indices
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(color='red', size=8),
        text=[str(i) for i in range(len(nodes))],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        title='Interactive Quadrilateral Mesh',
        xaxis=dict(scaleanchor='y'),
        yaxis=dict(scaleanchor='x'),
        width=700,
        height=700
    )

    if filename:
        fig.write_html(filename)
    if show:
        fig.show()
        

def plot_displacement_interactive(nodes, elems, U, scale=1.0, show=True, filename=None):
    """
    Interactive plot of vertical displacement (w) for a quadrilateral mesh.

    Parameters
    ----------
    nodes : ndarray (N,2)
        Node coordinates (x,y)
    elems : list of tuples/lists
        Each element is (n0,n1,n2,n3)
    U : ndarray
        Global displacement vector
    scale : float
        Amplification factor for visualization
    show : bool
        If True, shows the interactive plot
    filename : str or None
        If provided, saves the plot to HTML file
    """
    nodes = np.asarray(nodes)
    x = nodes[:,0]
    y = nodes[:,1]

    ndof_per_node = 3
    w = U[0::ndof_per_node]  # κάθετη μετατόπιση w

    fig = go.Figure()

    # Χρωματισμός ανά στοιχείο (average w στα 4 nodes)
    w_elem = []
    for elem in elems:
        w_avg = np.mean(w[list(elem)])
        w_elem.append(w_avg)

    w_min = min(w_elem)
    w_max = max(w_elem)

    # Χρωματίζουμε τα quads ανά w
    for i, elem in enumerate(elems):
        quad_x = x[list(elem)] + w[list(elem)]*scale
        quad_y = y[list(elem)]
        w_val = w_elem[i]

        # normalize για color scale
        t = (w_val - w_min)/(w_max - w_min + 1e-12)
        color = f'rgba({int(255*t)},0,{int(255*(1-t))},0.6)'

        fig.add_trace(go.Scatter(
            x=quad_x.tolist() + [quad_x[0]],
            y=quad_y.tolist() + [quad_y[0]],
            mode='lines',
            line=dict(color='black'),
            fill='toself',
            fillcolor=color,
            showlegend=False
        ))

    # Nodes with w values
    for i, xi, yi, wi in zip(range(len(x)), x, y, w):
        fig.add_trace(go.Scatter(
            x=[xi + wi*scale],
            y=[yi],
            mode='markers+text',
            marker=dict(color='blue', size=6),
            text=[f"{wi:.3e}"],
            textposition='top center',
            showlegend=False
        ))

    fig.update_layout(
        title='Vertical displacement (w) - amplified',
        xaxis=dict(scaleanchor='y'),
        yaxis=dict(scaleanchor='x'),
        width=700,
        height=700
    )

    if filename:
        fig.write_html(filename)
    if show:
        fig.show()
        
def plot_deformed_shape(nodes, elements, U, scale=100, filename=None):
    """
    Plot undeformed and deformed quadrilateral mesh.

    Parameters
    ----------
    nodes : ndarray (N,2)
        Node coordinates (x,y)
    elements : list of tuples
        Each element = (n0, n1, n2, n3)
    U : ndarray
        Global displacement vector (assume 2 DOF per node: Ux, Uy)
    scale : float
        Amplification factor for visualization
    """

    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_nodes = nodes.shape[0]

    # reshape global vector U (3 DOF per node: w, θx, θy)
    U_reshaped = U.reshape((n_nodes, 3))

    # Χρήση θx, θy για 2D παραμόρφωση
    Uxy = np.zeros((n_nodes,2))
    Uxy[:,0] = U_reshaped[:,1]  # θx → x
    Uxy[:,1] = U_reshaped[:,2]  # θy → y

    # Παραμορφωμένα nodes
    nodes_def = nodes + Uxy * scale

    plt.figure()
    for elem in elements:
        plt.fill(nodes[elem,0], nodes[elem,1], 'b', alpha=0.2)
        plt.fill(nodes_def[elem,0], nodes_def[elem,1], 'r', alpha=0.5)

    plt.axis('equal')
    plt.grid(True)
    plt.legend(['Αρχική γεωμετρία', 'Παραμορφωμένη γεωμετρία'])
    plt.title('Deformed shape of structure')

    if filename:
        plt.savefig(filename)

    plt.show()
    
def plot_stress_map(data, filename=None, show=True):
    """
    Δημιουργία Scatter Plot (Heatmap) των τάσεων Von Mises.
    
    Parameters
    ----------
    data : ndarray
        Ο πίνακας δεδομένων από τον PostProcessor.
        Στήλες: [ElemID, PointID, x, y, sx, sy, txy, VonMises]
    filename : str, optional
    """
    # Ανάκτηση δεδομένων από τις στήλες
    # x είναι η στήλη 2, y η στήλη 3, VonMises η στήλη 7
    x = data[:, 2]
    y = data[:, 3]
    vm = data[:, 7] 

    plt.figure(figsize=(10, 8))
    
    # Scatter plot με χρώμα ανάλογα την τάση Von Mises
    # s=20 είναι το μέγεθος της κουκκίδας
    sc = plt.scatter(x, y, c=vm, cmap='jet', s=20, edgecolors='none')
    
    plt.colorbar(sc, label='Von Mises Stress (Pa)')
    plt.title('Von Mises Stress Distribution (Top Surface)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Stress map plot saved to {filename}")
    
    if show:
        plt.show()
    
    plt.close()
