import visdom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

class Visualize():
    def __init__(self, args, env='main'):
        self.vis = visdom.Visdom(env=env)
        self.n_imgs = 5

    def show_images(self, images, win=''):
        n_imgs = min(self.n_imgs, images.shape[0])
        # show_images = images[:n_imgs].expand(n_imgs, 3, images.size(2), images.size(3))
        self.vis.images(images[:n_imgs], win=win, opts=dict(title=win))

    def show_lines(self, iter, item, win=''):
        y = torch.Tensor([list(item.values())]).reshape(1, -1)
        x = torch.Tensor([[iter for _ in range(len(item))]]).reshape(1, -1)

        self.vis.line(Y=y,
                      X=x,
                      win=win,
                      opts=dict(title=win,
                                legend=list(item.keys()),
                                showlegend=True),
                      update='append' if iter != 0 else 'replace')

    def show_plots(self, plots, win='', cmap='hot'):
        n_imgs = min(self.n_imgs, plots.shape[0])

        h, w = plots.shape[-2:]
        plots_array = plots.cpu().numpy()

        fig, ax = plt.subplots(1, n_imgs)
        for i in range(n_imgs):
            ax[i].imshow(plots_array[i][0], cmap='seismic')
            # ax[i] = sns.heatmap(plots_array[i][0], cmap='seismic')
            ax[i].axis('off')

        # fig.colorbar(ax[-1])
        # plt.show()
        self.vis.matplot(fig, win=win, opts=dict(title=win))

        return

