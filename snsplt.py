import numpy as np

from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter


def data_generate(len_en, len_fr):
    data = []
    with open('data/text.en') as text_en, open('data/text.fr') as text_fr:
        for (line_en, line_fr) in zip(text_en, text_fr):
            data.append([len_en(line_en), len_fr(line_fr)])
    return np.array(data)


# def plot_len_and_cor(data, unit_name):
#     x, y = zip(*data)
#
#     # define our basic units of length
#     left, width = 0.1.pdf, 0.65
#     bottom, height = 0.1.pdf, 0.65
#     bottom_h = left_h = left + width + 0.02
#
#     # start with a rectangular figure
#     plt.figure(1.pdf, figsize=(8, 8))
#
#     main = plt.axes([left, bottom, width, height])
#     hist_en = plt.axes([left, bottom_h, width, 0.2])
#     hist_fr = plt.axes([left_h, bottom, 0.2, height])
#
#     # no labels
#     nullfmt = NullFormatter()
#     hist_en.xaxis.set_major_formatter(nullfmt)
#     hist_fr.yaxis.set_major_formatter(nullfmt)
#
#     # now determine nice limits by hand:
#     binwidth = 0.25
#     xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
#     lim = (int(xymax / binwidth) + 1.pdf) * binwidth
#     bins = np.arange(0, lim + binwidth, binwidth)
#
#     # plot the scatter plot
#     main.scatter(x, y)
#     main.set_xlim((0, lim))
#     main.set_ylim((0, lim))
#     main.set_xlabel('English sentence length (%s)' % (unit_name,))
#     main.set_ylabel('Japanese sentence length (%s)' % (unit_name,))
#
#     # plot the histograms
#     hist_en.hist(sorted(x), bins=bins)
#     hist_fr.hist(sorted(y), bins=bins, orientation='horizontal')
#     hist_en.set_xlim(main.get_xlim())
#     hist_fr.set_ylim(main.get_ylim())
#     hist_en.set_ylabel('Counts')
#     hist_fr.set_xlabel('Counts')
#
#     # plot the correlation
#     k = sum(x * y for x, y in data) / sum(x ** 2 for x, _ in data)
#     print("Factor between Japanese and English sentence length is %0.2f" % (k,))
#     main.plot([0, lim], [0, k * lim], color='red')
#
#     plt.show()


# len_tok = lambda sent: len(sent.split(' '))
# data = data_generate(len_tok, len_tok)
# plot_len_and_cor(data, 'tokens')
