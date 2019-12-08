import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn import neighbors
import torch
import logging
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from matplotlib import cm
from pandas import DataFrame

# plt.style.use('/home/admin/jupyter/xdwang-paper.mplstyle')

def draw_tsne(extractor, classifier, src_data_loader, tgt_data_loader, res_dir, idx, title, separate=True):
    extractor.eval()
    classifier.eval()

    src_y = []
    src_y_pred = []
    src_embeddings = []
    TOTAL_COUNT = 1
    count = 0
    with torch.no_grad():
        for (images, labels) in src_data_loader:
            src_y.append(labels.numpy())

            images = images.cuda()
            labels = labels.squeeze_().cuda()

            embedding = extractor(images)
            preds = classifier(embedding)
            pred_cls = preds.data.max(1)[1]

            src_y_pred.append(pred_cls.cpu().numpy())
            src_embeddings.append(embedding.cpu())
            count += 1
            if count >= TOTAL_COUNT:
                break

    tgt_y = []
    tgt_y_pred = []
    tgt_embeddings = []
    count = 0
    with torch.no_grad():
        for (images, labels) in tgt_data_loader:
            tgt_y.append(labels.numpy())

            images = images.cuda()
            labels = labels.squeeze_().cuda()

            embedding = extractor(images)
            preds = classifier(embedding)
            pred_cls = preds.data.max(1)[1]

            tgt_y_pred.append(pred_cls.cpu().numpy())
            tgt_embeddings.append(embedding.cpu())
            count += 1
            if count >= TOTAL_COUNT:
                break


    src_embeddings = np.concatenate(src_embeddings)
    tgt_embeddings = np.concatenate(tgt_embeddings)
    embeddings = np.concatenate((src_embeddings, tgt_embeddings))
    print('before tsne: merge src {} and tgt {} into single array {}'.format(src_embeddings.shape, tgt_embeddings.shape, embeddings.shape))
    sk_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    embeddings_2d = sk_tsne.fit_transform(embeddings)
    src_embeddings_2d = embeddings_2d[:src_embeddings.shape[0]]
    tgt_embeddings_2d = embeddings_2d[src_embeddings.shape[0]:]
    print('after tsne: src_embeddings_2d {} and tgt_embeddings_2d {}'.format(src_embeddings_2d.shape, tgt_embeddings_2d.shape))


    # 1. real label
    src_y = np.concatenate(src_y)
    tgt_y = np.concatenate(tgt_y)
    print('src_y {}, tgt_y {}'.format(src_y.shape, tgt_y.shape))

    if separate:
    #    #with plt.style.context('xdwang-paper'):
    #    plt.figure(figsize=(5, 5))
    #    for i in range(len(np.unique(src_y))):
    #        data = src_embeddings_2d[src_y==i]
    #        plt.scatter(data[:, 0], data[:, 1], label=i, marker='.', s=10)
    #    plt.savefig('{}/{}_tsne_src_real.png'.format(res_dir, idx), dvi=300)
    #    print('save tsne_src_real to {}/{}_tsne_src_real.png'.format(res_dir, idx))
    #    plt.close()

    #    plt.figure(figsize=(5, 5))
    #    for i in range(len(np.unique(tgt_y))):
    #        data = tgt_embeddings_2d[tgt_y==i]
    #        plt.scatter(data[:, 0], data[:, 1], label=i, marker='.', s=10)
    #    plt.savefig('{}/{}_tsne_tgt_real.png'.format(res_dir, idx), dvi=300)
    #    print('save tsne_tgt_real to {}/{}_tsne_tgt_real.png'.format(res_dir, idx))
    #    plt.close()
    # else:
    #     with plt.style.context('xdwang-paper'):
    #         plt.figure(figsize=(5, 5))
    #         for i in range(len(np.unique(src_y))):
    #             data = src_embeddings_2d[src_y==i]
    #             # plt.scatter(data[:, 0], data[:, 1], label=i)
    #             c = cm.rainbow(int(255 * i / len(np.unique(src_y))))
    #             plt.text(data[:, 0], data[:, 1], i, backgroundcolor=c, fontsize=9)

    #         for i in range(len(np.unique(tgt_y))):
    #             data = tgt_embeddings_2d[tgt_y==i]
    #             # plt.scatter(data[:, 0], data[:, 1], label=i)
    #             c = cm.rainbow(int(255 * i / len(np.unique(src_y))))
    #             plt.text(data[:, 0], data[:, 1], i, fontsize=9, bbox=dict(boxstyle='circle', fc=c))

    #         plt.savefig('{}/{}_tsne_together.png'.format(res_dir, idx), dvi=300)
    #         print('save _tsne_together to {}/{}_tsne_together.png'.format(res_dir, idx))
    #         plt.close()
    #else:
        # with plt.style.context('xdwang-paper'):
        x_min = min(src_embeddings_2d[:, 0].min(), tgt_embeddings_2d[:, 0].min())
        x_max = max(src_embeddings_2d[:, 0].max(), tgt_embeddings_2d[:, 0].max())
        y_min = min(src_embeddings_2d[:, 1].min(), tgt_embeddings_2d[:, 1].min())
        y_max = max(src_embeddings_2d[:, 1].max(), tgt_embeddings_2d[:, 1].max())
        plt.figure(figsize=(10, 10))
        plt.title(title, fontsize=16)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        for data, label in zip(src_embeddings_2d, src_y):
            x = data[0]
            y = data[1]
            c = cm.rainbow(int(255 * label / len(np.unique(src_y))))
            plt.text(x, y, label, backgroundcolor=c, fontsize=9)
            # plt.text(x, y, label, fontsize=9)

        for data, label in zip(tgt_embeddings_2d, tgt_y):
            x = data[0]
            y = data[1]
            c = cm.rainbow(int(255 * label / len(np.unique(tgt_y))))
            plt.text(x, y, label, fontsize=9, bbox=dict(boxstyle='circle', fc=c))
            # plt.text(x, y, label, fontsize=9)

        plt.savefig('{}/{}_tsne_label.png'.format(res_dir, idx), dvi=300)
        print('save _tsne_together to {}/{}_tsne_together.png'.format(res_dir, idx))
        plt.close()

    # 画在一起，类用形状区别，域用颜色区别
    else:
        x_min = min(src_embeddings_2d[:, 0].min(), tgt_embeddings_2d[:, 0].min())
        x_max = max(src_embeddings_2d[:, 0].max(), tgt_embeddings_2d[:, 0].max())
        y_min = min(src_embeddings_2d[:, 1].min(), tgt_embeddings_2d[:, 1].min())
        y_max = max(src_embeddings_2d[:, 1].max(), tgt_embeddings_2d[:, 1].max())
        plt.figure(figsize=(10, 10))
        plt.title(title, fontsize=16)
        # plt.xticks(fontproperties = 'Times New Roman', size = 14)
        # plt.yticks(fontproperties = 'Times New Roman', size = 14)
        plt.xticks(size = 14)
        plt.yticks(size = 14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        source_color = 'red'
        target_color = 'blue'

        # 标记类型：星号，正方形，加号，倒三角
        marker_list = ['*', 's', '+', 'v']

        for data, label in zip(src_embeddings_2d, src_y):
            x = data[0]
            y = data[1]
            c = cm.rainbow(int(255 * label / len(np.unique(src_y))))
            plt.scatter(x, y, label=label, marker=marker_list[label], s=200, color=source_color)


        for data, label in zip(tgt_embeddings_2d, tgt_y):
            x = data[0]
            y = data[1]
            c = cm.rainbow(int(255 * label / len(np.unique(tgt_y))))
            plt.scatter(x, y, label=label, marker=marker_list[label], s=200, color=target_color)

        # plt.legend()
        plt.savefig('{}/{}_tsne_colors.png'.format(res_dir, idx), dvi=300)
        print('save _tsne_together to {}/{}_tsne_together.png'.format(res_dir, idx))
        plt.close()

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c][c] / df_cm[c].sum() )
        # sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
        # sum_lin.append( item_line[1].sum() )
    df_cm['Acc'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['Recall'] = sum_col

def insert_acc_recall(df_cm, tp_per_class):
    accs = []
    recalls = []
    for c in range(df_cm.shape[0]):
        tp = df_cm[c][c]
        positive = df_cm[:, c].sum()
        accs.append(round(tp/tp_per_class, 2))
        recalls.append(round(tp/positive, 2))
    recalls.append(np.nan)
    #print('accs: {}'.format(accs))
    #print('recalls: {}'.format(recalls))
    
    accs = np.array(accs)
    accs = accs[:, np.newaxis]
    recalls = np.array(recalls)
    recalls = recalls[np.newaxis, :]

    df_cm = np.concatenate((df_cm, accs), axis=1)
    df_cm = np.concatenate((df_cm, recalls), axis=0)
    #print(df_cm)
    return df_cm

def do_draw_confusion_matrix(y, y_pred, res_dir, epoch, title):
    #sns.set_style("whitegrid")
    classes = np.unique(y)
    confusion_matrix_result = confusion_matrix(y, y_pred, labels=classes)
    print('draw_confusion_matrix, y {}, y_pred {}'.format(y.shape, y_pred.shape))
    tp_per_class = int(len(y) / len(classes))
    print('len(classes): {}, num per class: {}'.format(len(classes), tp_per_class))
    if (len(classes) == 3):
        # paderborn c3
        labellist = ['H', 'OR', 'IR']
        xlabellist = ['H', 'OR', 'IR', 'Acc']
        ylabellist = ['H', 'OR', 'IR', 'Recall']
    elif (len(classes) == 15):
        # paderborn 15
        labellist = ['K001', 'K002', 'K003', 'K004', 'K005', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI04', 'KI14', 'KI16', 'KI18', 'KI21']
        xlabellist = ['K001', 'K002', 'K003', 'K004', 'K005', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI04', 'KI14', 'KI16', 'KI18', 'KI21', 'Acc']
        ylabellist = ['K001', 'K002', 'K003', 'K004', 'K005', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI04', 'KI14', 'KI16', 'KI18', 'KI21', 'Recall']
    elif (len(classes) == 10):
        # cwru deonly
        labellist = ['H', 'IR7', 'IR14','IR21','OR7','OR14','OR21','B7','B14','B21']
        xlabellist = ['H', 'IR7', 'IR14','IR21','OR7','OR14','OR21','B7','B14','B21', 'Acc']
        ylabellist = ['H', 'IR7', 'IR14','IR21','OR7','OR14','OR21','B7','B14','B21', 'Recall']
    elif (len(classes) == 4):
        # cwru defe
        labellist = ['B', 'IR', 'H', 'OR',]
        xlabellist =['B', 'IR', 'H', 'OR', 'Acc']
        ylabellist =['B', 'IR', 'H', 'OR', 'Recall']
    else:
        labellist = classes
        xlabellist = classes
        ylabellist = classes

    print(type(confusion_matrix_result))
    confusion_matrix_result = insert_acc_recall(confusion_matrix_result, tp_per_class=tp_per_class)

    fig, ax = plt.subplots()
    #sns.set_style("whitegrid")
    # sns.heatmap(confusion_matrix_result, xticklabels=classes, yticklabels=classes, annot=True, fmt="d", annot_kws={"size": 10})
    sns.heatmap(confusion_matrix_result, xticklabels=xlabellist, yticklabels=ylabellist, cmap=plt.cm.Blues, annot=True, fmt="g", cbar=False, annot_kws={"size": 10})
    plt.grid(True, which='minor', linestyle='-')
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes)
    # plt.yticks(tick_marks, classes)
    plt.title(title, fontsize=16)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.tight_layout()
    plt.savefig('{}/confusion_{}.png'.format(res_dir, epoch), dvi=300)
    plt.close()


def draw_confusion_matrix(extractor, classifier, data_loader, res_dir, epoch, title):
    extractor.eval()
    classifier.eval()

    y = []
    y_pred = []
    with torch.no_grad():
        for (features, labels) in data_loader:
            y.append(labels)

            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            preds = classifier(extractor(features))
            pred_cls = preds.data.max(1)[1]

            y_pred.append(pred_cls.cpu())

    y = np.concatenate(y)
    y_pred = np.concatenate(y_pred)

    do_draw_confusion_matrix(y, y_pred, res_dir, epoch, title)
