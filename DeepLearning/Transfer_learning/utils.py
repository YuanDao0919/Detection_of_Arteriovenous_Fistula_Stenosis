import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, recall_score, confusion_matrix

# 生成并保存详细分类报告
def generate_classification_report(y_true, y_pred, save_path=None):
    report = classification_report(y_true, y_pred, digits=4)
    
    # 计算每个类别的F1分数
    class_f1 = f1_score(y_true, y_pred, average=None)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # 转换为二分类标签：类别0为0，类别1和2合并为1
    binary_y_true = [0 if label == 0 else 1 for label in y_true]
    binary_y_pred = [0 if pred == 0 else 1 for pred in y_pred]
    binary_f1 = f1_score(binary_y_true, binary_y_pred, pos_label=1)
    binary_sensitivity = recall_score(binary_y_true, binary_y_pred, pos_label=1)
    cm_binary = confusion_matrix(binary_y_true, binary_y_pred)
    if cm_binary.shape == (2, 2):
        tn, fp, fn, tp = cm_binary.ravel()
        binary_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        binary_specificity = 0.0

    report_str = f"多分类分类报告:\n{report}\n"
    report_str += f"各类别F1分数: {class_f1}\n"
    report_str += f"宏平均F1分数: {macro_f1:.4f}\n\n"
    report_str += "二分类指标 (0 vs. {1,2}):\n"
    report_str += f"  Sensitivity (Recall): {binary_sensitivity:.4f}\n"
    report_str += f"  Specificity: {binary_specificity:.4f}\n"
    report_str += f"  F1 Score: {binary_f1:.4f}\n"

    print(report_str)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)

    return report_str

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, save_path=None, class_names=None):
    if class_names is None:
        class_names = ["类别0", "类别1", "类别2"]
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    for i, class_name in enumerate(class_names):
        plt.text(-0.5, i + 0.5, f'召回率: {recall[i]:.2f}', 
                horizontalalignment='center', verticalalignment='center')
        plt.text(len(class_names) + 0.5, i + 0.5, f'精确率: {precision[i]:.2f}', 
                horizontalalignment='center', verticalalignment='center')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 绘制整体混淆矩阵（用于交叉验证结果）
def plot_overall_confusion_matrix(all_labels, all_predictions, save_path=None, class_names=None):
    if class_names is None:
        class_names = ["类别0", "类别1", "类别2"]
    
    cm = confusion_matrix(all_labels, all_predictions)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Overall Confusion Matrix (Cross-Validation)')
    
    for i, class_name in enumerate(class_names):
        plt.text(-0.5, i + 0.5, f'Recall: {recall[i]:.2f}',
                horizontalalignment='center', verticalalignment='center')
        plt.text(len(class_names) + 0.5, i + 0.5, f'Precision: {precision[i]:.2f}',
                horizontalalignment='center', verticalalignment='center')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return cm, precision, recall

# 可视化特征重要性
def visualize_feature_importance(model, data_loader, save_path=None):
    model.eval()
    device = next(model.parameters()).device
    features_by_class = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for batch in data_loader:
            signals = batch['signal'].to(device)
            labels = batch['label'].cpu().numpy()
            _, _, features = model(signals)
            features = features.cpu().numpy()

            for i, label in enumerate(labels):
                features_by_class[label].append(features[i])

    avg_features = {}
    for label, feat_list in features_by_class.items():
        if feat_list:
            avg_features[label] = np.mean(np.stack(feat_list), axis=0)

    plt.figure(figsize=(12, 6))
    for label, feat in avg_features.items():
        plt.plot(feat, label=f'类别 {label}')

    plt.xlabel('特征维度')
    plt.ylabel('平均激活值')
    plt.title('各类别特征重要性')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
# 添加torch导入，避免特征重要性可视化时出现引用错误
import torch 