

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from distance_metrics import load_gower_matrix, euclidean_distance



BLUE =  "#1f77b4"
ORANGE = "#ff7f0e"



def plot_distance(distances, title='Distance',dataset_name=''):
    min_sorted = sorted(distances)
    plt.figure(figsize=(10, 10))
    plt.plot(min_sorted, label='Min', color=ORANGE, linestyle='-', alpha=1)
    plt.title("Distance to Closest Record - " + dataset_name,fontsize=20)
    plt.xlabel("Sorted Row Index (by Min Distance)",fontsize=16)
    plt.ylabel(title,fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)   
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_min_distances(min_d, q1, median, q3, max_d, title='Data',dataset_name=''):
    sort_idx = np.argsort(min_d)
    x = np.arange(len(min_d))

    min_sorted = sorted(min_d)
    q1_sorted = sorted(q1)
    median_sorted = sorted(median)
    q3_sorted = sorted(q3)
    max_sorted = sorted(max_d)

    plt.figure(figsize=(10, 10))

    # Fill IQR (shaded area between Q1 and Q3)
    plt.fill_between(x, q1_sorted, q3_sorted, alpha=0.2, color=BLUE, label='IQR (Q1–Q3)')

    # Plot lines using same color with different styles
    plt.plot(median_sorted, label='Median (Q2)', color=BLUE, linewidth=2)
    plt.plot(q1_sorted, label='Q1 (25%)', color=BLUE, linestyle='--', linewidth=1)
    plt.plot(q3_sorted, label='Q3 (75%)', color=BLUE, linestyle='--', linewidth=1)
    plt.plot(min_sorted, label='Min', color=ORANGE, linestyle='-', alpha=1)
    plt.plot(max_sorted, label='Max', color='black', linestyle='-', alpha=1)

    plt.title(f"Distance to Other Records - " + dataset_name, fontsize=20)
    plt.xlabel("Sorted Row Index (by Min Distance)",fontsize=16)
    plt.ylabel(title, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)   
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)  # Increase font size by 50% (default is 8)   plt.tight_layout()
    plt.show()



def compute_distance_and_plot(df1, df2=None, categorical_columns=[], dataset_name='', hybrid=False):
    remove_self = df2 is None

    # Helper function to compute 5-number summary 
    def compute_summary(dist_matrix):
        min_ = dist_matrix.min(axis=1)
        q1 = np.quantile(dist_matrix, 0.25, axis=1)
        median = np.quantile(dist_matrix, 0.5, axis=1)
        q3 = np.quantile(dist_matrix, 0.75, axis=1)
        max_ = dist_matrix.max(axis=1)
        return min_, q1, median, q3, max_


    gower_distance_1 = load_gower_matrix(df1, df2, alpha=1, remove_self=remove_self)
    g1_min, g1_q1, g1_median, g1_q3, g1_max = compute_summary(gower_distance_1)
    plot_min_distances(g1_min, g1_q1, g1_median, g1_q3, g1_max, "HGD-1", dataset_name)


    
    gower_distance_2 = load_gower_matrix(df1, df2, alpha=2, remove_self=remove_self)
    g2_min, g2_q1, g2_median, g2_q3, g2_max = compute_summary(gower_distance_2)
    plot_min_distances(g2_min, g2_q1, g2_median, g2_q3, g2_max, "HGD-2", dataset_name)

    if hybrid: return gower_distance_1, gower_distance_2, None

    euclidean_dist = euclidean_distance(df1, df2, categorical_columns=categorical_columns, remove_self=remove_self)
    euc_min, euc_q1, euc_median, euc_q3, euc_max = compute_summary(euclidean_dist)
    plot_min_distances(euc_min, euc_q1, euc_median, euc_q3, euc_max, "Euclidean Distance", dataset_name)
    

    return gower_distance_1, gower_distance_2, euclidean_dist
