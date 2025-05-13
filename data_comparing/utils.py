import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.rcParams.update({
        'axes.labelsize': 18,      # xlabel and ylabel
        'xtick.labelsize': 15,     # x-axis tick labels
        'ytick.labelsize': 15,     # y-axis tick labels
        'axes.titlesize': 18,      # subplot title
        'legend.fontsize': 15,     # legend
        'font.size': 15            # fallback default
    })

def compare_dataframes(
    df: pd.DataFrame,
    syn: pd.DataFrame,
    nominal_values: dict,
    categorical_columns: list = None,
    columns: list = None
):
    """
    Compare each column in df and syn using:
    - Histogram if the column has continuous data
    - Bar chart if the column has categorical data

    Mixed columns are plotted alone; one-type columns (cat or cont only) are plotted two per row.
    """
    categorical_columns = categorical_columns or []
    columns_to_plot = columns if columns is not None else df.columns
    one_part_buffer = []

    num_bins = 50  # Number of histogram bins

    def plot_one_part_pair(pair, num_bins):
        plt.figure(figsize=(14, 5))
        for i, (col, cat_vals, cont_vals, is_cat) in enumerate(pair):
            plt.subplot(1, 2, i + 1)
            if is_cat:
                # Sort original categories by frequency
                orig_counts_sorted = cat_vals[0].value_counts()
                all_categories = orig_counts_sorted.index.tolist()

                # Add synthetic-only categories at the end
                extra_categories = [cat for cat in cat_vals[1].unique() if cat not in all_categories]
                all_categories.extend(extra_categories)

                orig_counts = cat_vals[0].value_counts().reindex(all_categories, fill_value=0)
                syn_counts = cat_vals[1].value_counts().reindex(all_categories, fill_value=0)
                width = 0.35
                x = np.arange(len(all_categories))
                plt.bar(x - width / 2, orig_counts.values, width=width, label='Original', edgecolor='black')
                plt.bar(x + width / 2, syn_counts.values, width=width, label='Synthetic', edgecolor='black')
                plt.xticks(ticks=x, labels=all_categories, rotation=45, fontsize=15)
                plt.yticks(fontsize=15)
                plt.title(f'{col} (Categorical)', fontsize=18)
                plt.xlabel('Category', fontsize=18)
                plt.ylabel('Count', fontsize=18)
                plt.legend(fontsize=15)
            else:
                bin_edges = np.histogram_bin_edges(np.concatenate([cont_vals[0], cont_vals[1]]), bins=num_bins)
                plt.hist(cont_vals[0], bins=bin_edges, alpha=0.6, label='Original', edgecolor='black')
                plt.hist(cont_vals[1], bins=bin_edges, alpha=0.6, label='Synthetic', edgecolor='black')
                plt.title(f'{col} (Continuous)', fontsize=18)
                plt.xlabel('Value', fontsize=18)
                plt.ylabel('Frequency', fontsize=18)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.legend(fontsize=15)
        plt.tight_layout()
        plt.show()

    for col in columns_to_plot:
        original = df[col]
        synthetic = syn[col]

        if col in categorical_columns:
            cat_orig = original.fillna('NaN').astype(str)
            cat_syn = synthetic.fillna('NaN').astype(str)
            cont_orig = cont_syn = pd.Series(dtype=float)
        else:
            mixed_values = nominal_values.get(col, [])

            def split_series(series, mixed_values):
                is_categorical = series.apply(
                    lambda x: any((pd.isna(val) and pd.isna(x)) or (str(x) == str(val)) for val in mixed_values)
                )
                categorical = series[is_categorical].fillna('NaN')
                continuous = pd.to_numeric(series[~is_categorical], errors='coerce').dropna().astype(float)
                return categorical, continuous

            cat_orig, cont_orig = split_series(original, mixed_values)
            cat_syn, cont_syn = split_series(synthetic, mixed_values)

        has_cont = not cont_orig.empty or not cont_syn.empty
        has_cat = not cat_orig.empty or not cat_syn.empty

        # Case 1: Mixed → full-row layout
        if has_cat and has_cont:
            plt.figure(figsize=(14, 6))
            # Continuous plot
            plt.subplot(1, 2, 1)
            bin_edges = np.histogram_bin_edges(np.concatenate([cont_orig, cont_syn]), bins=num_bins)
            plt.hist(cont_orig, bins=bin_edges, alpha=0.6, label='Original', edgecolor='black')
            plt.hist(cont_syn, bins=bin_edges, alpha=0.6, label='Synthetic', edgecolor='black')
            plt.title(f'{col} (Continuous)', fontsize=18)
            plt.xlabel('Value', fontsize=15)
            plt.ylabel('Frequency', fontsize=18)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15)

            # Categorical plot
            plt.subplot(1, 2, 2)
            # Sort original categories by frequency
            orig_counts_sorted = cat_vals[0].value_counts()
            all_categories = orig_counts_sorted.index.tolist()

            # Add synthetic-only categories at the end
            extra_categories = [cat for cat in cat_vals[1].unique() if cat not in all_categories]
            all_categories.extend(extra_categories)

            orig_counts = cat_orig.value_counts().reindex(all_categories, fill_value=0)
            syn_counts = cat_syn.value_counts().reindex(all_categories, fill_value=0)
            width = 0.35
            x = np.arange(len(all_categories))
            plt.bar(x - width / 2, orig_counts.values, width=width, label='Original', edgecolor='black')
            plt.bar(x + width / 2, syn_counts.values, width=width, label='Synthetic', edgecolor='black')
            plt.xticks(ticks=x, labels=all_categories, rotation=45, fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(f'{col} (Categorical)', fontsize=18)
            plt.xlabel('Category', fontsize=18)
            plt.ylabel('Count', fontsize=18)
            plt.legend(fontsize=15)

            plt.tight_layout()
            plt.show()

        # Case 2: Only categorical or continuous → buffer for side-by-side plotting
        elif has_cat:
            one_part_buffer.append((col, (cat_orig, cat_syn), None, True))
        elif has_cont:
            one_part_buffer.append((col, None, (cont_orig, cont_syn), False))

        if len(one_part_buffer) == 2:
            plot_one_part_pair(one_part_buffer, num_bins)
            one_part_buffer = []

    # Plot final leftover column if odd number
    if len(one_part_buffer) == 1:
        plot_one_part_pair([one_part_buffer[0]], num_bins)




def plot_single_dataframe(
    df: pd.DataFrame,
    nominal_values: dict,
    categorical_columns: list = None,
    columns: list = None
):
    """
    Plot distribution of each column in a DataFrame:
    - Bar chart for categorical columns
    - Histogram for continuous columns
    - Both if column has mixed types (based on nominal_values)

    Parameters:
    - df: DataFrame to plot
    - nominal_values: dict mapping column names to known nominal values (can include np.nan)
    - categorical_columns: list of column names to treat as fully categorical
    - columns: optional list of columns to include (defaults to all)
    """
    categorical_columns = categorical_columns or []
    columns_to_plot = columns if columns is not None else df.columns
    one_part_buffer = []

    bins = 50

    def plot_one_part_pair(pair,bins):
        plt.figure(figsize=(14, 5))
        for i, (col, cat_vals, cont_vals, is_cat) in enumerate(pair):
            plt.subplot(1, 2, i + 1)
            if is_cat:
                counts = cat_vals.value_counts()
                x = np.arange(len(counts))
                plt.bar(x, counts.values, edgecolor='black')
                plt.xticks(ticks=x, labels=counts.index, rotation=45, fontsize=15)
                plt.yticks(fontsize=15)
                plt.title(f'{col} (Categorical)', fontsize=18)
                plt.xlabel('Category', fontsize=18)
                plt.ylabel('Count', fontsize=18)
            else:
                bins = np.histogram_bin_edges(cont_vals, bins=bins)
                plt.hist(cont_vals, bins=bins, alpha=0.7, edgecolor='black')
                plt.title(f'{col} (Continuous)', fontsize=18)
                plt.xlabel('Value', fontsize=18)
                plt.ylabel('Frequency', fontsize=18)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

    for col in columns_to_plot:
        series = df[col]

        if col in categorical_columns:
            cat_vals = series.fillna('NaN').astype(str)
            cont_vals = pd.Series(dtype=float)
        else:
            mixed = nominal_values.get(col, [])

            def split_series(s, mixed_values):
                is_cat = s.apply(lambda x: any((pd.isna(v) and pd.isna(x)) or (str(x) == str(v)) for v in mixed_values))
                cat = s[is_cat].fillna('NaN')
                cont = pd.to_numeric(s[~is_cat], errors='coerce').dropna().astype(float)
                return cat, cont

            cat_vals, cont_vals = split_series(series, mixed)

        has_cat = not cat_vals.empty
        has_cont = not cont_vals.empty

        if has_cat and has_cont:
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            bins = np.histogram_bin_edges(cont_vals, bins=bins)
            plt.hist(cont_vals, bins=bins, alpha=0.7, edgecolor='black')
            plt.title(f'{col} (Continuous)', fontsize=18)
            plt.xlabel('Value', fontsize=18)
            plt.ylabel('Frequency', fontsize=18)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.subplot(1, 2, 2)
            counts = cat_vals.value_counts()
            x = np.arange(len(counts))
            plt.bar(x, counts.values, edgecolor='black')
            plt.xticks(ticks=x, labels=counts.index, rotation=45, fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(f'{col} (Categorical)', fontsize=18)
            plt.xlabel('Category', fontsize=18)
            plt.ylabel('Count', fontsize=18)

            plt.tight_layout()
            plt.show()

        elif has_cat:
            one_part_buffer.append((col, cat_vals, None, True))
        elif has_cont:
            one_part_buffer.append((col, None, cont_vals, False))

        if len(one_part_buffer) == 2:
            plot_one_part_pair(one_part_buffer,bins)
            one_part_buffer = []

    if len(one_part_buffer) == 1:
        plot_one_part_pair([one_part_buffer[0]],bins)




def plot_fidelity(fidelity_list, bar_labels=None):
   
    num_datasets = len(fidelity_list)
    # Generate default labels if none are provided
    if bar_labels is None:
        bar_labels = [f'Distance {i}' for i in range(num_datasets)]
    else:
        assert len(bar_labels) == num_datasets, "Length of bar_labels must match number of datasets."

    df_all = fidelity_lists_to_feature_list(fidelity_list, bar_labels)

    # Function to plot a single metric type (JSD or WD)
    def plot_metric(df_subset, metric_name):
        x = np.arange(len(df_subset))
        bar_width = 0.8 / num_datasets

        fig, ax = plt.subplots(figsize=(14, 10))
        for i in range(num_datasets):
            ax.bar(x + (i - num_datasets / 2 + 0.5) * bar_width,
                   df_subset[bar_labels[i]],
                   bar_width,
                   label=bar_labels[i])

        ax.set_xlabel('Feature')
        ax.set_ylabel(f'{metric_name} Distance')
        ax.set_title(f'Comparison of {metric_name} Distances Across Features')
        ax.set_xticks(x)
        ax.set_xticklabels(df_subset['labels'], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # Split and plot both metrics
    for metric in df_all['metrics'].unique():
        subset = df_all[df_all['metrics'] == metric]
        plot_metric(subset.reset_index(drop=True), metric)




def fidelity_lists_to_feature_list(fidelity_list,bar_labels=None):
    distance_frames = [entry[1] for entry in fidelity_list]
    num_datasets = len(distance_frames)

    # Generate default labels if none are provided
    if bar_labels is None:
        bar_labels = [f'Distance {i}' for i in range(num_datasets)]
    else:
        assert len(bar_labels) == num_datasets, "Length of bar_labels must match number of datasets."

    # Assume all have the same "Column" and "Metric"
    labels = distance_frames[0]["Column"]
    weights = distance_frames[0]["Weight"]
    metrics = distance_frames[0]["Metric"]

    # Build combined dataframe
    data = {
        'labels': labels,
        'weights': weights,
        'metrics': metrics,
    }
    for i, df in enumerate(distance_frames):
        data[bar_labels[i]] = df["Distance"]

    return pd.DataFrame(data)

def fidelity_lists_to_feature_summary(fidelity_list, bar_labels=None):

    distance_frames = [entry[0] for entry in fidelity_list]
    num_datasets = len(distance_frames)

    # Generate default labels if none are provided
    if bar_labels is None:
        bar_labels = [f'Distance {i}' for i in range(num_datasets)]
    else:
        assert len(bar_labels) == num_datasets, "Length of bar_labels must match number of datasets."

    data = {
    'metrics': distance_frames[0]["Metric"]
    }
    for i, df in enumerate(distance_frames):
        data[bar_labels[i]] = df["Weighted_Avg"]

    return pd.DataFrame(data)


def fidelity_lists_to_diff_corr_summary(fidelity_list, bar_labels=None):

    distance_frames = [entry[2] for entry in fidelity_list]
    num_datasets = len(distance_frames)

    # Generate default labels if none are provided
    if bar_labels is None:
        bar_labels = [f'Distance {i}' for i in range(num_datasets)]
    else:
        assert len(bar_labels) == num_datasets, "Length of bar_labels must match number of datasets."

    data = {
    'Diff. Corr': ["Forbeanious","MAE"]
    }
    forbeanious = [entry["forbenious"] for entry in distance_frames]
    mae = [entry["mae"] for entry in distance_frames]
    data.update({label: [forbeanious[i], mae[i]] for i, label in enumerate(bar_labels)})
    mae = [entry["mae"] for entry in distance_frames]
    

    return pd.DataFrame(data)
    


def privacy_lists_to_summary(privacy_list, bar_labels=None):

    num_datasets = len(privacy_list)
    if bar_labels is None:
        bar_labels = [f'Distance {i}' for i in range(num_datasets)]
    else:
        assert len(bar_labels) == num_datasets, "Length of bar_labels must match number of datasets."
    
    distances = [entry[0] for entry in privacy_list]
    distances = pd.concat(distances, axis=1).transpose().rename_axis("Mapping").reset_index()
    distances.insert(0, 'Privacy', [bar_labels[i//3] if i % 3 == 0 else "" for i in range(len(distances))])

    NNAAs = [entry[1] for entry in privacy_list]
    df_NNAAs = pd.DataFrame({'Privacy': bar_labels, 'NNAA': NNAAs})
    return distances, df_NNAAs


def utility_list_to_summary(utility_list, bar_labels=None):
    
    num_datasets = len(utility_list)

    if bar_labels is None:
        bar_labels = [f'Distance {i}' for i in range(num_datasets)]
    else:
        assert len(bar_labels) == num_datasets, "Length of bar_labels must match number of datasets."

    reals = [entry[0] for entry in utility_list]
    fakes = [entry[1][0] for entry in utility_list]

    

    reals_df = pd.concat(reals, axis=0)
    fakes_df = pd.concat(fakes, axis=0)
    full = pd.concat([reals_df, fakes_df], axis=1)
    full = full.reset_index().rename(columns={'index': 'Method'})

    measures_count = reals_df.shape[1] 

    dp_descriptions1 = [bar_labels[i//num_datasets] if i % num_datasets == 0 else "" for i in range(full.shape[0])]

    full.insert(0, 'DP', dp_descriptions1)

    reals_df2 = pd.concat(reals, axis=1)
    fakes_df2 = pd.concat(fakes, axis=1)
    differnce = reals_df2 - fakes_df2
    differnce_mean = differnce.mean(axis=0)

    
    dp_descriptions2 = [bar_labels[i//measures_count] if i % measures_count == 0 else "" for i in range(len(differnce_mean))]

    differnce_df = pd.DataFrame({
        'DP': dp_descriptions2,
        'Metric': differnce_mean.index,
        'Mean Difference': differnce_mean.values,
    })
    return full, differnce_df