import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from cycler import cycler

cc = (cycler(marker=['o', 'x', '*', 'v']) *
      (cycler(markerfacecolor=list('rgbyckm')) + cycler(markeredgecolor=list('rgbyckm')))
      )
import matplotlib

matplotlib.rcParams.update({'font.size': 20})


def heatmap_reuse_vs_size(all_heatmap_data, all_reuse_bins, min_sizes,
                          all_heatmap_data_entr_reuse, all_entr_bins, all_reuse_hists,
                          cm_name='Reds', size_compress_factor=1, size_tick_stop=10, entr_step_size=3,
                          white_val=np.array([1, 1, 1, 1]),
                          num_color_ticks=5):
    for k, heatmap_data in all_heatmap_data.items():
        plt.figure("H" + str(k))
        orig_data = heatmap_data[:-1, :]
        max_size = min_sizes[k] + orig_data.shape[1]
        if size_compress_factor != 1:
            newdata = np.zeros((orig_data.shape[0], int(orig_data.shape[1] / size_compress_factor)))
            for col in range(newdata.shape[1]):
                newdata[:, col] = orig_data[:, col * size_compress_factor:(col + 1) * size_compress_factor].sum(axis=1)
            for row in range(newdata.shape[0]):
                sum_this_thow = newdata[row, :].sum()
                newdata[row, :] = newdata[row, :] / sum_this_thow
        else:
            newdata = orig_data

        # ---------------------- THE REUSE vs SIZE HISTORGRAM ------
        # -------------------------------------------------------------------------------------
        # we make a copy of the colorschme, but making sure the value 0 is asigned to white_val
        colors = plt.cm.__dict__[cm_name](np.linspace(0, 1, 256))
        colors[0, :] = white_val
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

        plt.subplot(1,2,1)
        plt.imshow(newdata, cmap=cmap, interpolation='none', aspect='auto')
        reuse_bins = all_reuse_bins[k]
        plt.yticks(list(range(len(reuse_bins) - 1)), ["%.2f" % a for a in reuse_bins[:-1]])
        plt.ylabel("Mean decomposition reusability ")
        xticks = [10 * int(i * size_tick_stop / 10)
                  for i in range(max_size)
                  if i * size_tick_stop < newdata.shape[1]]
        xtick_labels = [10 * int((min_sizes[k] + i * size_tick_stop * size_compress_factor) / 10)
                        for i in range(len(xticks))]
        xtick_labels[0] = min_sizes[k]
        plt.xticks(xticks, xtick_labels)
        plt.xlabel("PBB size [number of elements] ")
        plt.gca().invert_yaxis()

        #The color bar
        color_tick_step = np.round(newdata.max() / num_color_ticks,decimals=2)
        colorticks = np.round([i * color_tick_step for i in range(1,num_color_ticks+1)],decimals=2)
        clb = plt.colorbar(ticks=colorticks, orientation='horizontal',shrink=0.7)
        clb.ax.set_xlabel('Frequency of PBB size')

        # ---------------------- THE REUSE vs ENTROPY HISTORGRAM ------
        # -------------------------------------------------------------------------------------
        # we make a copy of the colorschme, but making sure the value 0 is asigned to white_val
        colors = plt.cm.__dict__['Blues'](np.linspace(0, 1, 256))
        colors[0, :] = white_val
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

        plt.subplot(1,2,2)
        entr_bins = all_entr_bins[k]
        plt.imshow(all_heatmap_data_entr_reuse[k][:-1, :], cmap=cmap, interpolation='none', aspect='auto')
        plt.gca().invert_yaxis()
        plt.yticks([])
        xticks = [entr_step_size*i for i in range(int(len(entr_bins)/entr_step_size))]
        xtick_labels = ["{:10.2f}".format(entr_bins[entr_step_size*i])
                        for i in range(int(len(entr_bins)/entr_step_size))]
        plt.xticks(xticks, xtick_labels)
        plt.xlabel("Entropy of the distribution of PBB sizes [bits]")
        clb = plt.colorbar(orientation='horizontal',shrink=0.7)
        clb.ax.set_xlabel('Frequency of entropy')

        plt.subplots_adjust(wspace=0.01, right=0.97, left=0.07, top=0.96, bottom=0.01)



def scatter_reuse_vs_muDistance(all_reuses_and_delta_mus, plot_each=False):
    plt.figure("Delta" + str(0))
    ax = plt.gca()
    ax.set_prop_cycle(cc)
    for k, reuses_and_delta_mus in all_reuses_and_delta_mus.items():
        fig_numbers = [0, k] if plot_each else [0]
        for fig_n in fig_numbers:
            plt.figure("Delta" + str(fig_n))
            all_reuses = [x[0] for x in reuses_and_delta_mus if x[1] > 0]
            all_deltas = [x[1] for x in reuses_and_delta_mus if x[1] > 0]
            if fig_n==0:
                plt.plot(all_reuses, all_deltas, label="k=" + str(k), ls='None')
            else:
                plt.plot(all_reuses, all_deltas, 'b.')
            plt.xlabel("Mean Decomposition Reusability")
            plt.ylabel("Distance between size distribution maxima")
    plt.figure("Delta" + str(0))
    plt.legend(loc=9, ncol=6)


def scatter_reuse_vs_entropiesSizes(all_reuses_and_size_entropies, plot_each):
    plt.figure("Entr" + str(0))
    ax = plt.gca()
    ax.set_prop_cycle(cc)
    for k, reuses_and_size_entropies in all_reuses_and_size_entropies.items():
        fig_numbers = [k,0] if plot_each else [0]
        for fig_n in fig_numbers:
            plt.figure("Entr" + str(fig_n))
            all_reuses = np.array([x[0] for x in reuses_and_size_entropies])
            all_entropies = np.array([x[1] for x in reuses_and_size_entropies])
            if fig_n==0:
                plt.plot(all_reuses, all_entropies, label="k=" + str(k), ls='None')
            else:
                plt.plot(all_reuses, all_entropies, 'b.')
            plt.xlabel("Mean Decomposition Reusability")
            plt.ylabel("Entropy of the distribution of PBB sizes")
    plt.figure("Entr" + str(0))
    plt.legend(loc=9, ncol=6)
    #ax = plt.gca()
    #ax.set_prop_cycle(cc)
