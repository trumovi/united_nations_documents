import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from text import Text
import os
import numpy as np
import matplotlib.pyplot as plt


def make_plot(text_object: Text, prediction: np.ndarray, name):
    # Calculate estimated r(n,k) using optimized parameters
    r_nk_est = prediction
    # Round estimated values to integers for frequency counts
    r_nk_est_rounded = np.round(r_nk_est, 0)

    # Calculate exact word counts per frequency
    # Difference between consecutive r(n,k) values gives count of words at each frequency
    exact_counts = (r_nk_est_rounded[:-1] - r_nk_est_rounded[1:]).astype(int)
    exact_counts = np.append(exact_counts, r_nk_est_rounded[-1])  # Add last frequency count

    # Reconstruct rank-frequency distribution
    ranks_new = []
    freq_new = []
    current_rank = 1

    # For each frequency level, create corresponding ranks
    for freq, count in zip(range(1, len(exact_counts) + 1), exact_counts.astype(int)):
        ranks_new.extend(range(current_rank, current_rank + count))  # Assign ranks
        freq_new.extend([freq] * count)  # Repeat frequency for each word
        current_rank += count  # Increment rank counter

    plt.figure(figsize=(10, 6))

    # Plot: Rank-frequency distribution
    plt.loglog(text_object.ranks_empirical, text_object.frequencies_empirical,
                 'o', markersize=2., label="Empirical", color="blue")
    plt.loglog(ranks_new, freq_new[::-1],
                 label="Estimated", color="red")

    plt.xlabel("Rank",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    #plt.show()
    plt.savefig(name + ".png")




for subdir, _, files in os.walk("un_texts"):
    for file in files:
        #files_to_check = ("_1_eng.txt")
        no_check =  ("ch.txt", "arabic.txt", "eng.txt")
        no_st = ("doc_12", "doc_10", "doc_11")
        if file.endswith("sp.txt") and not file.startswith(no_st):
            filepath = os.path.join(subdir, file)
            print(file)
            text_analysis = Text(filepath)

            print('Text size:', text_analysis.size)
            print('Number of unique words:', text_analysis.different_words_count)
            print('Word frequencies in text:', text_analysis.frequencies_empirical)
            print('Possible frequencies:', text_analysis.frequencies_range)
            print('Empirical r_nk:', text_analysis.r_nk_empirical)
            print('f_0:', text_analysis.f_0)
            print('ranks_empirical:', text_analysis.ranks_empirical)

            some_a = 1.2
            some_q = 5.2
            some_m = 10000

            text_analysis.r_nk_approx(some_a, some_q, some_m)

            text_analysis.powell_optimization()

            print("\n")
            print(f'Final parameters: ')
            if text_analysis.f_0 < 3:
                print('Single optimization used')
                print('sigma =', text_analysis.approx_std_full)
            else:
                print(f'Optimization 1:\nalpha_1 = {text_analysis.estimated_parameters_1[0]},\n'
                      f'q_1 = {text_analysis.estimated_parameters_1[1]},\nM_1 = {text_analysis.estimated_parameters_1[2]}')
                print('sigma_1 =', text_analysis.approx_std_1)
                print(f'Optimization 2:\nalpha_2 = {text_analysis.estimated_parameters_2[0]},\n'
                      f'q_2 = {text_analysis.estimated_parameters_2[1]},\nM_2 = {text_analysis.estimated_parameters_2[2]}')
                print('sigma_2 =', text_analysis.approx_std_2)
                print('Final std sigma =', text_analysis.full_approx_std)
            name = file.rstrip(".txt")
            make_plot(text_analysis, text_analysis.r_nk_prediction_full, name)
            print("-----------------------------------------------------------------\n")


