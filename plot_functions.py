import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import matplotlib.colors as mcolors

from cycler import cycler

cc = (cycler(marker=['o', 'x', '*', 'v']) *
      (cycler(markerfacecolor=list('rgbyckm')) + cycler(markeredgecolor=list('rgbyckm')))
      )
import matplotlib

matplotlib.rcParams.update({'font.size': 19})

def compress_data(orig_data, size_compress_factor):
    newdata = np.zeros((orig_data.shape[0], int(orig_data.shape[1] / size_compress_factor)))
    for col in range(newdata.shape[1]):
        newdata[:, col] = orig_data[:, col * size_compress_factor:(col + 1) * size_compress_factor].sum(axis=1)
    for row in range(newdata.shape[0]):
        sum_this_thow = newdata[row, :].sum()
        newdata[row, :] = newdata[row, :] / sum_this_thow
    return newdata

def heatmap_triptych(all_heatmap_data, all_reuse_bins, min_sizes, all_reuse_hists,
                     all_large_reuse_hists,all_small_reuse_hists,
                     cm_name='Reds', size_compress_factor=1, size_tick_stop=30, entr_step_size=3,
                     white_val=np.array([1, 1, 1, 1]),
                     num_color_ticks=5, division_pos=120
                     ):

    for k, heatmap_data in all_heatmap_data.items():

        plt.figure("H" + str(k))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])
        axL = plt.subplot(gs[0])
        axC = plt.subplot(gs[1])
        axR = plt.subplot(gs[2])

        reuse_bins = all_reuse_bins[k]
        orig_data = heatmap_data[:-1, :]
        max_size = min_sizes[k] + orig_data.shape[1]
        if size_compress_factor != 1:
            newdata = compress_data(orig_data, size_compress_factor)
        else:
            newdata = orig_data

        # Central piece:  size vs mean reusability
        plt.sca(axC)
        make_size_mean_reuse_heatmap(cm_name, white_val, newdata, reuse_bins,
                                     size_tick_stop, max_size, min_sizes[k], size_compress_factor, num_color_ticks,
                                     showylabel_and_ticks=False)
        plt.vlines(division_pos, 0, len(all_reuse_bins[k])-2, colors='g', linestyles='dashed')
        plt.text(division_pos + 2, 2, "Large PBBs", color='g', fontsize = 17)
        plt.text(division_pos - 37, 2, "Small PBBs", color='g', fontsize = 17)
        plt.title("Whole decomposition")

        #side pieces: individual module reuse vs mean reusability of decompo
        for subpn, subax, data_dict,tit in [(1, axL, all_small_reuse_hists[k], "Small PBBs"),
                                        (3, axR, all_large_reuse_hists[k], "Large PBBs")]:
            n =  [x.shape for x in data_dict.values()][0][0]
            data = np.zeros((1+max(data_dict.keys()), n  ) )
            for rn,ve in data_dict.items():
                data[rn,:] = ve/ve.sum()

            plt.sca(subax)
            colors = plt.cm.__dict__['Blues'](np.linspace(0, 1, 256))
            colors[0, :] = white_val
            cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

            reuse_hists = all_reuse_hists[k]
            plt.imshow(data[:, 1:], cmap=cmap, interpolation='none', aspect='auto', vmin=0, vmax=1)
            if subpn == 1:
                plt.yticks([x - 0.5 for x in range(len(reuse_bins))],
                           ["%.2f" % a for a in reuse_bins[:]])
                plt.ylabel("Mean decomposition reusability ")
            elif subpn == 3:
                plt.yticks(list(range(len(reuse_hists))), ["  " + str(int(a)) + "" for a in reuse_hists])
                plt.ylabel("Number of decompositions per reusability range")
                plt.gca().yaxis.tick_right()
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position('none')
            plt.xlabel("Individual PBB Reusability")
            plt.xticks([5*i for i in range(n) if 5*i <= n], [str(1+5*i) for i in range(n) if 5*i <= n])

            plt.gca().invert_yaxis()
            clb = plt.colorbar(orientation='horizontal', shrink=0.7, ticks=[0,0.5,1],  pad = 0.1)
            clb.ax.set_xlabel('Frequency of\nPBB reusability')
            plt.title(tit)
            plt.gca()

        plt.subplots_adjust(wspace=0.01, right=0.93, left=0.06, top=0.93, bottom=0.0)






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
            newdata = compress_data(orig_data, size_compress_factor)
        else:
            newdata = orig_data


        plt.subplot(1, 2, 1)
        make_size_mean_reuse_heatmap(cm_name, white_val, newdata, all_reuse_bins[k],
                                     size_tick_stop, max_size, min_sizes[k], size_compress_factor, num_color_ticks)
        plt.hlines

        # ---------------------- THE REUSE vs ENTROPY HISTORGRAM ------
        # -------------------------------------------------------------------------------------
        # we make a copy of the colorschme, but making sure the value 0 is asigned to white_val
        plt.subplot(1,2,2)
        colors = plt.cm.__dict__['Blues'](np.linspace(0, 1, 256))
        colors[0, :] = white_val
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

        reuse_hists = all_reuse_hists[k]
        entr_bins = all_entr_bins[k]
        plt.imshow(all_heatmap_data_entr_reuse[k][:-1, :], cmap=cmap, interpolation='none', aspect='auto')
        plt.gca().invert_yaxis()
        plt.gca().yaxis.tick_right()
        plt.yticks(list(range(len(reuse_hists))), ["  "+str(int(a))+"" for a in reuse_hists])
        plt.ylabel("Number of decompositions per reusability range")
        plt.gca().yaxis.set_label_position("right")
        xticks = [entr_step_size*i for i in range(int(len(entr_bins)/entr_step_size))]
        xtick_labels = ["{:10.2f}".format(entr_bins[entr_step_size*i])
                        for i in range(int(len(entr_bins)/entr_step_size))]
        plt.xticks(xticks, xtick_labels)
        plt.xlabel("Entropy of the distribution of PBB sizes [bits]")
        clb = plt.colorbar(orientation='horizontal',shrink=0.7, pad = 0.1)
        clb.ax.set_xlabel('Frequency of entropy')

        plt.subplots_adjust(wspace=0.01, right=0.925, left=0.07, top=0.96, bottom=0.01)


def make_size_mean_reuse_heatmap(cm_name, white_val, newdata, reuse_bins,
                                 size_tick_stop, max_size, min_sizes, size_compress_factor, num_color_ticks,
                                 showylabel_and_ticks=True):
    # ---------------------- THE REUSE vs SIZE HISTORGRAM ------
    # -------------------------------------------------------------------------------------
    # we make a copy of the colorschme, but making sure the value 0 is asigned to white_val
    colors = plt.cm.__dict__[cm_name](np.linspace(0, 1, 256))
    colors[0, :] = white_val
    cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

    plt.imshow(newdata, cmap=cmap, interpolation='none', aspect='auto')
    if showylabel_and_ticks:
        plt.yticks([x - 0.5 for x in range(len(reuse_bins))],
                   ["%.2f" % a for a in reuse_bins[:]])
        plt.ylabel("Mean decomposition reusability ")
    else:
        plt.yticks([])
    xticks = [10 * int(i * size_tick_stop / 10)
              for i in range(max_size)
              if i * size_tick_stop < newdata.shape[1]]
    xtick_labels = [10 * int((min_sizes + i * size_tick_stop * size_compress_factor) / 10)
                    for i in range(len(xticks))]
    xtick_labels[0] = min_sizes
    plt.xticks(xticks, xtick_labels)
    plt.xlabel("PBB size [number of elements] ")
    plt.gca().invert_yaxis()

    # The color bar
    color_tick_step = np.round(newdata.max() / num_color_ticks, decimals=2)
    colorticks = np.round([i * color_tick_step for i in range(1, num_color_ticks + 1)], decimals=2)
    clb = plt.colorbar(ticks=colorticks, orientation='horizontal', shrink=0.7, pad = 0.1)
    clb.ax.set_xlabel('Frequency of PBB size')


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
