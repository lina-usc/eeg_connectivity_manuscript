import numpy as np
import pandas as pd
import scipy.stats


def compute_bootstrap_mse_corr(overall_estimate_dict, overall_ground_truth_dict, confounder_list, all_methods):
    mse_dict = {}
    corr_coef_dict = {}

    for confounder in confounder_list:
        confounder_mse_dict = {}
        confounder_corr_dict = {}

        for method in all_methods:
            connections = overall_estimate_dict[confounder][method].keys()
            connection_mse_dict = {}
            connection_corr_dict = {}

            for connection in connections:
                mse_list = []
                corr_coef_list = []

                for _ in range(1000):
                    estimates = np.array(overall_estimate_dict[confounder][method][connection])
                    index = np.random.choice(range(100), 100, replace=True)
                    bootstrap_estimates = estimates[index]

                    ground_truth = np.array(overall_ground_truth_dict[confounder][method][connection])
                    bootstrap_ground = ground_truth[index]

                    mse_list.append(np.sum((bootstrap_estimates - bootstrap_ground) ** 2) / 100)
                    corr = scipy.stats.spearmanr(bootstrap_estimates, bootstrap_ground)[0]
                    corr_coef_list.append(0.0 if np.isnan(corr) else corr)

                connection_mse_dict[connection] = mse_list
                connection_corr_dict[connection] = corr_coef_list

            confounder_mse_dict[method] = connection_mse_dict
            confounder_corr_dict[method] = connection_corr_dict

        mse_dict[confounder] = confounder_mse_dict
        corr_coef_dict[confounder] = confounder_corr_dict

    return mse_dict, corr_coef_dict


def compute_ci_dict(mse_dict, corr_coef_dict, overall_estimate_dict, confounder_list, comparison_pairs):
    ci_dict_mse = {}
    ci_dict_corr = {}

    for confounder in confounder_list:
        confounder_dict_mse = {}
        confounder_dict_corr = {}

        for pair in comparison_pairs:
            connections = overall_estimate_dict[confounder][pair[0]].keys()
            pair_dict_mse = {}
            pair_dict_corr = {}

            for connection in connections:
                g1 = np.array(mse_dict[confounder][pair[0]][connection])
                g1.sort()
                g2 = np.array(mse_dict[confounder][pair[1]][connection])
                g2.sort()
                diff = g2 - g1
                diff.sort()
                pair_dict_mse[connection] = (np.round(diff[24], 3), np.round(diff[974], 3))

                g1 = np.array(corr_coef_dict[confounder][pair[0]][connection])
                g2 = np.array(corr_coef_dict[confounder][pair[1]][connection])
                diff = g2 - g1
                diff.sort()
                pair_dict_corr[connection] = (np.round(diff[24], 3), np.round(diff[974], 3))

            if pair == ('direct_directed_transfer_function', 'generalized_partial_directed_coherence'):
                pair = ('dDTF', 'gPDC')
            elif pair == ('direct_directed_transfer_function', 'pairwise_spectral_granger_prediction'):
                pair = ('dDTF', 'pSGP')
            elif pair == ('generalized_partial_directed_coherence', 'pairwise_spectral_granger_prediction'):
                pair = ('gPDC', 'pSGP')

            confounder_dict_mse[pair] = pair_dict_mse
            confounder_dict_corr[pair] = pair_dict_corr

        ci_dict_mse[confounder] = confounder_dict_mse
        ci_dict_corr[confounder] = confounder_dict_corr

    return ci_dict_mse, ci_dict_corr


def build_ci_dataframe(ci_dict_mse, confounder_list, slice_range):
    df_list = []
    for confounder in confounder_list:
        df1 = pd.DataFrame(list(ci_dict_mse[confounder].keys())[slice_range])
        df1['Pairs'] = df1[0] + "-" + df1[1]
        df1 = df1.drop(0, axis=1).drop(1, axis=1)
        df2 = pd.DataFrame(list(ci_dict_mse[confounder].values())[slice_range])
        df3 = df1.join(df2, how="right")
        df3.insert(0, "Confounder", "")
        df3.loc[df3.index[0], "Confounder"] = confounder
        df_list.append(df3)
    return df_list
