import matplotlib.pyplot as plt
import itertools
import numpy as np
import functools
import operator
import pandas as pd
import os
def draw_train_history_plot(train_history, picture_name:str, location:str):
    """
    繪製訓練歷史圖表並保存為圖片。

    Args:
        train_history (History): 訓練歷史物件，包含了訓練過程中的損失和準確度等信息。
        picture_name (str): 要保存的圖片名稱。
        location (str): 保存圖片的路徑。

    Returns:
        None
    """
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    n = range(1, len(train_history.history['loss']) + 1)
    plt.plot(n, train_history.history["loss"], label="train_loss")
    plt.plot(n, train_history.history["val_loss"], label="val_loss")
    plt.plot(n, train_history.history["accuracy"], label="train_acc")
    plt.plot(n, train_history.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(f'{location+picture_name}.png')

def plot_confusion_matrix(picture_name: str,
                          location:str,
                          cm,
                          classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    繪製混淆矩陣並保存為圖片。

    Args:
        picture_name (str): 要保存的圖片名稱。
        cm (numpy array): 混淆矩陣。
        classes (list): 類別名稱列表。
        normalize (bool): 是否進行正規化。
        cmap: 繪圖的顏色映射。
        location (str): 保存圖片的路徑。

    Returns:
        None
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_path = os.path.join(location, f'{picture_name}.png')
    plt.savefig(save_path)
    plt.close()  # 关闭图像，防止内存泄漏


def lithology_plot(dfRow,
                   classifier_number:str,
                   picture_name: str,
                   location:str,
                   start_depth=None,
                   end_depth=None,
                   save_ax=None):
    """
        繪製地層圖。

        Args:
            dfRow (DataFrame): 包含地層數據的資料框。
            picture_name (str): 圖片名稱。
            location (str): 圖片保存位置。
            classifier_number (str): 繪圖顏色標籤
            start_depth (float, optional): 起始深度。默認為 None。
            end_depth (float, optional): 結束深度。默認為 None。
            save_ax (str, optional): 指定要單獨保存的子圖名稱。默認為 None。

        Returns:
            None

        """
    lithology_numbers_2 = {0: {'lith': 'mud', 'lith_num': 0, 'color': '#ffc081'},  # 標籤0：橘色
                              1: {'lith': 'Gravel', 'lith_num': 1, 'color': '#ccecff'}}  # 標籤1： 藍色

    lithology_numbers_3 = {0: {'lith': 'mud', 'lith_num': 0, 'color': '#ffc081'},  # 標籤0：橘色
                             1: {'lith': 'coarseSand', 'lith_num': 1, 'color': '#dce490'},  # 標籤1：綠色
                             2: {'lith': 'Gravel', 'lith_num': 2, 'color': '#ccecff'}}  # 標籤2： 藍色

    lithology_numbers_4 = {0: {'lith': 'mud', 'lith_num': 0, 'color': '#ffc081'},  # 標籤0：橘色
                             1: {'lith': 'fineSand', 'lith_num': 1, 'color': '#8A2BE2'},  # 標籤1：紫色
                             2: {'lith': 'coarseSand', 'lith_num': 2, 'color': '#dce490'}, #標籤2:綠色
                             3: {'lith': 'Gravel', 'lith_num': 3, 'color': '#ccecff'}}  # 標籤3： 藍色
    lithology_dict = {'2':lithology_numbers_2,
                      '3':lithology_numbers_3,
                      '4':lithology_numbers_4}

    well = pd.DataFrame()
    well["N16"] = functools.reduce(operator.iconcat, dfRow["N16"], [])
    well["N64"] = functools.reduce(operator.iconcat, dfRow["N64"], [])
    well["SP"] = functools.reduce(operator.iconcat, dfRow["SP"], [])
    well["GAMMA"] = functools.reduce(operator.iconcat, dfRow["GAMMA"], [])
    well["depth"] = functools.reduce(operator.iconcat, dfRow["Depth"], [])
    well["ANS"] = functools.reduce(operator.iconcat, dfRow["ANS"], [])
    well["predict"] = functools.reduce(operator.iconcat, dfRow["Prediction"], [])

    # 過濾深度範圍
    if start_depth is not None:
        well = well[well['depth'] >= start_depth]
    if end_depth is not None:
        well = well[well['depth'] <= end_depth]

    top_depth = min(well["depth"])
    bottom_depth = max(well["depth"])

    fig, ax = plt.subplots(figsize=(18, 10))
    scale = (bottom_depth - top_depth) / 30

    # Set up the plot axes
    ax0 = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid((1, 6), (0, 1), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 6), (0, 2), rowspan=1, colspan=1, sharey=ax1)
    ax3 = plt.subplot2grid((1, 6), (0, 3), rowspan=1, colspan=1, sharey=ax1)
    ax4 = plt.subplot2grid((1, 6), (0, 4), rowspan=1, colspan=1, sharey=ax1)
    ax5 = plt.subplot2grid((1, 6), (0, 5), rowspan=1, colspan=1, sharey=ax1)

    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax00 = ax0.twiny()
    ax00.xaxis.set_visible(False)
    ax10 = ax1.twiny()
    ax10.xaxis.set_visible(False)
    ax11 = ax2.twiny()
    ax11.xaxis.set_visible(False)
    ax13 = ax4.twiny()
    ax13.xaxis.set_visible(False)
    ax14 = ax5.twiny()
    ax14.xaxis.set_visible(False)

    # SP track
    ax0.plot(well["SP"], well['depth'], color="green", linewidth=0.5)
    ax0.set_xlabel("SP")
    ax0.xaxis.label.set_color("green")
    ax0.set_xlim(min(well["SP"]), max(well["SP"]))
    ax0.set_ylabel("Depth (m)")
    ax0.tick_params(axis='x', colors="green")
    ax0.spines["top"].set_edgecolor("green")
    ax0.title.set_color('green')

    # Gamma Ray track
    ax1.plot(well["GAMMA"], well['depth'], color="green", linewidth=0.5)
    ax1.set_xlabel("Gamma")
    ax1.xaxis.label.set_color("green")
    ax1.set_xlim(min(well["GAMMA"]), max(well["GAMMA"]))
    ax1.tick_params(axis='x', colors="green")
    ax1.spines["top"].set_edgecolor("green")

    # Density track
    ax2.plot(well["N64"], well['depth'], color="red", linewidth=0.5)
    ax2.set_xlabel("N64")
    ax2.set_xlim(min(well["N64"]), max(well["N64"]))
    ax2.xaxis.label.set_color("red")
    ax2.tick_params(axis='x', colors="red")
    ax2.spines["top"].set_edgecolor("red")

    # Neutron track
    ax3.plot(well["N16"], well['depth'], color="blue", linewidth=0.5)
    ax3.set_xlabel('N16')
    ax3.set_xlim(min(well["N16"]), max(well["N16"]))
    ax3.xaxis.label.set_color("blue")
    ax3.tick_params(axis='x', colors="blue")
    ax3.spines["top"].set_position(("axes", 1.08))
    ax3.spines["top"].set_visible(True)
    ax3.spines["top"].set_edgecolor("blue")

    # Lithology track # true label
    ax4.plot()
    ax4.set_xlabel("True Label")
    ax4.set_xlim(0, 1)

    depth = list(well["depth"])
    class1 = list(well["ANS"])
    for d in range(len(depth) - 1):
        if round(depth[d + 1] - depth[d], 2) == 0.1:
            ax4.axhspan(depth[d], depth[d + 1], facecolor=lithology_dict[classifier_number][class1[d]]['color'])
        else:
            ax4.axhspan(depth[d], depth[d] + 0.1, facecolor=lithology_dict[classifier_number][class1[d]]['color'])
    ax4.set_xticks([0, 1])

    # Lithology track # predict label
    ax5.plot()
    ax5.set_xlabel("Predict Label")
    ax5.set_xlim(0, 1)

    class1 = list(well["predict"])
    for d in range(len(depth) - 1):
        if round(depth[d + 1] - depth[d], 2) == 0.1:
            ax5.axhspan(depth[d], depth[d + 1], facecolor=lithology_dict[classifier_number][class1[d]]['color'])
        else:
            ax5.axhspan(depth[d], depth[d] + 0.1, facecolor=lithology_dict[classifier_number][class1[d]]['color'])
    ax5.set_xticks([0, 1])

    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    # 確保所有子圖的深度範圍一致
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.set_ylim(bottom_depth, top_depth)
        # ax.invert_yaxis()  # 確保深度從上到下遞增
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        plt.setp(ax.get_yticklabels(), visible=False)
    # 保存指定的子圖
    if save_ax is not None:
        save_axes = {
            'ax0': ax0, 'ax1': ax1, 'ax2': ax2, 'ax3': ax3, 'ax4': ax4, 'ax5': ax5
        }
        if save_ax in save_axes:
            ax_to_save = save_axes[save_ax]
            extent = ax_to_save.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{save_ax}.png', bbox_inches=extent)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    plt.savefig(f'{location+picture_name}.png')

__all__ = ['draw_train_history_plot', 'plot_confusion_matrix', 'lithology_plot']