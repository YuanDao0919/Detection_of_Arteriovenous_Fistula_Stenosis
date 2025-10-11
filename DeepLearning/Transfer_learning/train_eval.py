import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import pandas as pd
import seaborn as sns

from data import AVFDataset, set_seed
from model import Encoder, ImprovedAVFClassifier
from utils import generate_classification_report, plot_confusion_matrix, plot_overall_confusion_matrix, visualize_feature_importance

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练编码器
def load_pretrained_model(model_path):
    print(f"加载预训练模型: {model_path}")
    
    # 创建与预训练模型相同结构的编码器
    encoder = Encoder(input_size=600, hidden_size=128, num_layers=2, dropout=0.1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("成功加载检查点文件")
        
        # 尝试加载模型状态
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            try:
                # 筛选出编码器相关的参数
                encoder_dict = {}
                for k, v in checkpoint['model_state_dict'].items():
                    if k.startswith('encoder.'):
                        encoder_dict[k.replace('encoder.', '')] = v
                
                if encoder_dict:
                    # 尝试加载编码器参数
                    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_dict, strict=False)
                    print(f"成功加载编码器模型，缺失键: {len(missing_keys)}，未预期键: {len(unexpected_keys)}")
                else:
                    print("编码器参数未找到，尝试直接加载...")
                    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception as e:
                print(f"加载模型状态字典失败: {e}")
                print("尝试直接加载检查点文件...")
                encoder.load_state_dict(checkpoint, strict=False)
        else:
            # 尝试直接加载checkpoint作为模型状态
            try:
                encoder.load_state_dict(checkpoint, strict=False)
                print("使用checkpoint直接作为模型状态加载成功")
            except Exception as e:
                print(f"直接加载失败: {e}")
                print("警告: 使用随机初始化的编码器")
        
        return encoder
        
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        print("警告: 使用随机初始化的编码器")
        return encoder

# 评估模型函数
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            signals = batch['signal'].to(device)
            labels = batch['label'].to(device)

            main_outputs, aux_outputs, _ = model(signals)

            if criterion:
                main_loss = criterion(main_outputs, labels)
                aux_loss = criterion(aux_outputs, labels)
                loss = main_loss + 0.2 * aux_loss
                total_loss += loss.item() * signals.size(0)

            _, preds = torch.max(main_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 多分类指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    class_f1 = f1_score(all_labels, all_preds, average=None)
    class_precision = precision_score(all_labels, all_preds, average=None)
    class_recall = recall_score(all_labels, all_preds, average=None)

    # 二分类指标：将类别1和2合并为阳性（1），类别0作为阴性（0）
    binary_labels = [0 if label == 0 else 1 for label in all_labels]
    binary_preds = [0 if pred == 0 else 1 for pred in all_preds]
    binary_f1 = f1_score(binary_labels, binary_preds, pos_label=1)
    binary_sensitivity = recall_score(binary_labels, binary_preds, pos_label=1)  # Sensitivity即为阳性召回率
    # 计算特异性：TN / (TN + FP)
    cm_binary = confusion_matrix(binary_labels, binary_preds)
    if cm_binary.shape == (2, 2):
        tn, fp, fn, tp = cm_binary.ravel()
        binary_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        binary_specificity = 0.0

    result = {
        'accuracy': acc,
        'f1_score': macro_f1,
        'precision': precision,
        'recall': recall,
        'class_f1': class_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'predictions': all_preds,
        'ground_truth': all_labels,
        'binary_f1': binary_f1,
        'binary_sensitivity': binary_sensitivity,
        'binary_specificity': binary_specificity,
    }

    if criterion:
        result['loss'] = total_loss / len(data_loader.dataset)

    return result

# 训练函数
def train_transfer_model(model, train_loader, val_loader, num_epochs=100, initial_lr=0.001, 
                         save_dir="./avf_transfer_results_no_augmentation"):
    os.makedirs(save_dir, exist_ok=True)
    best_model_save_path = os.path.join(save_dir, "avf_classifier_best.pth")
    model = model.to(device)

    # 不使用类别权重
    criterion = nn.CrossEntropyLoss()
    print("训练不使用类别权重")

    # 初始冻结编码器
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': initial_lr},
        {'params': model.encoder.parameters(), 'lr': initial_lr * 0.1}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_f1_macro = 0.0
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_f1_scores = []

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        model.update_epoch(epoch, num_epochs)
        model.unfreeze_encoder()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            signals = batch['signal'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            main_outputs, aux_outputs, _ = model(signals, training_progress=epoch/num_epochs)
            main_loss = criterion(main_outputs, labels)
            aux_loss = criterion(aux_outputs, labels)
            loss = main_loss + 0.2 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(main_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)

        scheduler.step()

        val_result = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_result['loss'] if 'loss' in val_result else 0)
        val_f1_scores.append(val_result['f1_score'])

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_result.get('loss', 0):.4f} | Val Acc: {val_result['accuracy']:.4f} | Val Macro-F1: {val_result['f1_score']:.4f}")
        print(f"Val Precision: {val_result['precision']:.4f} | Val Recall: {val_result['recall']:.4f}")
        print(f"每类F1分数: 类别0: {val_result['class_f1'][0]:.4f}, 类别1: {val_result['class_f1'][1]:.4f}, 类别2: {val_result['class_f1'][2]:.4f}")
        print(f"二分类指标: Sensitivity: {val_result['binary_sensitivity']:.4f}, Specificity: {val_result['binary_specificity']:.4f}, Binary F1: {val_result['binary_f1']:.4f}")

        if val_result['accuracy'] > best_val_acc or (val_result['accuracy'] == best_val_acc and val_result['f1_score'] > best_val_f1_macro):
            best_val_acc = val_result['accuracy']
            best_val_f1_macro = val_result['f1_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1_score': best_val_f1_macro,
                'accuracy': best_val_acc,
                'precision': val_result['precision'],
                'recall': val_result['recall'],
                'class_f1': val_result['class_f1'].tolist(),
                'class_precision': val_result['class_precision'].tolist(),
                'class_recall': val_result['class_recall'].tolist(),
                'binary_f1': val_result['binary_f1'],
                'binary_sensitivity': val_result['binary_sensitivity'],
                'binary_specificity': val_result['binary_specificity']
            }, best_model_save_path)
            print(f"↑↑↑ 新的最佳模型已保存 (宏平均F1: {best_val_f1_macro:.4f}, Acc: {best_val_acc:.4f}, Precision: {val_result['precision']:.4f}, Recall: {val_result['recall']:.4f}) ↑↑↑")

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='训练损失')
            plt.plot(val_losses, label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(val_f1_scores, label='验证宏平均F1')
            plt.xlabel('Epoch')
            plt.ylabel('Macro F1-Score')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"learning_curves_epoch_{epoch+1}.png"))
            plt.close()

    print("\n训练完成！加载最佳模型进行最终评估...")
    best_checkpoint = torch.load(best_model_save_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    final_val_result = evaluate_model(model, val_loader)
    print(f"最佳模型验证集性能 - Accuracy: {final_val_result['accuracy']:.4f}, 宏平均F1: {final_val_result['f1_score']:.4f}")
    print(f"Precision: {final_val_result['precision']:.4f}, Recall: {final_val_result['recall']:.4f}")
    print(f"每类F1分数: 类别0: {final_val_result['class_f1'][0]:.4f}, 类别1: {final_val_result['class_f1'][1]:.4f}, 类别2: {final_val_result['class_f1'][2]:.4f}")
    print(f"二分类指标: Sensitivity: {final_val_result['binary_sensitivity']:.4f}, Specificity: {final_val_result['binary_specificity']:.4f}, Binary F1: {final_val_result['binary_f1']:.4f}")

    plot_confusion_matrix(
        final_val_result['ground_truth'], 
        final_val_result['predictions'],
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
        class_names=["类别0", "类别1", "类别2"]
    )

    visualize_feature_importance(
        model, 
        val_loader,
        save_path=os.path.join(save_dir, "feature_importance.png")
    )

    generate_classification_report(
        final_val_result['ground_truth'],
        final_val_result['predictions'],
        save_path=os.path.join(save_dir, "classification_report.txt")
    )

    print(f"所有结果已保存至 {save_dir}")
    return model, final_val_result

# 主函数
def main():
    set_seed(42)
    data_folder = "/home/swucar/cyz/1 pos"
    pretrained_model_path = "/home/swucar/cyz/bloodpressure/330codeL1L2/ppg_pretrain_models/ppg_pretrain_20250330_214849_encoder_best.pth"
    save_dir = "/home/swucar/cyz/bloodpressure/329code/duibishiyanjieg/fusion_claude筛选"
    os.makedirs(save_dir, exist_ok=True)

    print(f"正在加载预训练模型: {pretrained_model_path}")
    pretrained_model = load_pretrained_model(pretrained_model_path)

    print(f"正在加载数据集: {data_folder}")
    dataset = AVFDataset(
        data_folder=data_folder,
        apply_filter=True
    )

    class_counts = np.bincount([int(label) for label in dataset.labels])
    print(f"类别分布: {class_counts}")
    print("注意：本次训练不使用数据增强和类别权重")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    all_fold_predictions = []
    all_fold_ground_truth = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.labels)):
        print(f"\n{'='*50}")
        print(f"开始训练第 {fold+1}/{n_splits} 折")
        print(f"{'='*50}")

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model = ImprovedAVFClassifier(
            pretrained_encoder=pretrained_model,
            hidden_dim=128,
            num_classes=3,
            dropout_rate=0.5
        )

        fold_save_dir = os.path.join(save_dir, f"fold_{fold+1}")
        model, fold_result = train_transfer_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            initial_lr=0.001,
            save_dir=fold_save_dir
        )

        results.append({
            'fold': fold+1,
            'accuracy': fold_result['accuracy'],
            'macro_f1_score': fold_result['f1_score'],
            'precision': fold_result['precision'],
            'recall': fold_result['recall'],
            'class_f1_scores': fold_result['class_f1'].tolist(),
            'class_precision': fold_result['class_precision'].tolist(),
            'class_recall': fold_result['class_recall'].tolist(),
            'binary_f1': fold_result['binary_f1'],
            'binary_sensitivity': fold_result['binary_sensitivity'],
            'binary_specificity': fold_result['binary_specificity']
        })

        all_fold_predictions.extend(fold_result['predictions'])
        all_fold_ground_truth.extend(fold_result['ground_truth'])

    print("\n交叉验证结果汇总:")
    accuracies = [r['accuracy'] for r in results]
    macro_f1_scores = [r['macro_f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    binary_f1s = [r['binary_f1'] for r in results]
    binary_sensitivities = [r['binary_sensitivity'] for r in results]
    binary_specificities = [r['binary_specificity'] for r in results]

    print(f"平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"平均宏平均F1分数: {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}")
    print(f"平均精确率: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"平均召回率: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"平均二分类F1: {np.mean(binary_f1s):.4f} ± {np.std(binary_f1s):.4f}")
    print(f"平均敏感性: {np.mean(binary_sensitivities):.4f} ± {np.std(binary_sensitivities):.4f}")
    print(f"平均特异性: {np.mean(binary_specificities):.4f} ± {np.std(binary_specificities):.4f}")

    class_f1_matrix = np.array([r['class_f1_scores'] for r in results])
    class_precision_matrix = np.array([r['class_precision'] for r in results])
    class_recall_matrix = np.array([r['class_recall'] for r in results])
    mean_class_f1 = np.mean(class_f1_matrix, axis=0)
    std_class_f1 = np.std(class_f1_matrix, axis=0)
    mean_class_precision = np.mean(class_precision_matrix, axis=0)
    std_class_precision = np.std(class_precision_matrix, axis=0)
    mean_class_recall = np.mean(class_recall_matrix, axis=0)
    std_class_recall = np.std(class_recall_matrix, axis=0)

    for i, (mean_f1, std_f1, mean_prec, std_prec, mean_rec, std_rec) in enumerate(
        zip(mean_class_f1, std_class_f1, mean_class_precision, std_class_precision, mean_class_recall, std_class_recall)):
        print(f"类别 {i}:")
        print(f"  平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  平均精确率: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"  平均召回率: {mean_rec:.4f} ± {std_rec:.4f}")

    cm, overall_precision, overall_recall = plot_overall_confusion_matrix(
        all_fold_ground_truth,
        all_fold_predictions,
        save_path=os.path.join(save_dir, "confusion_matrix_all_folds.png"),
        class_names=["类别0", "类别1", "类别2"]
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Class 0", "Class 1", "Class 2"], 
                yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(save_dir, "cross_validation_results.csv"), index=False)

    summary = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_macro_f1': np.mean(macro_f1_scores),
        'std_macro_f1': np.std(macro_f1_scores),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_binary_f1': np.mean(binary_f1s),
        'std_binary_f1': np.std(binary_f1s),
        'mean_binary_sensitivity': np.mean(binary_sensitivities),
        'std_binary_sensitivity': np.std(binary_sensitivities),
        'mean_binary_specificity': np.mean(binary_specificities),
        'std_binary_specificity': np.std(binary_specificities),
        'mean_class_f1': mean_class_f1.tolist(),
        'std_class_f1': std_class_f1.tolist(),
        'mean_class_precision': mean_class_precision.tolist(),
        'std_class_precision': std_class_precision.tolist(),
        'mean_class_recall': mean_class_recall.tolist(),
        'std_class_recall': std_class_recall.tolist(),
        'overall_confusion_matrix': cm.tolist()
    }

    with open(os.path.join(save_dir, "summary_results.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n完成！所有模型、混淆矩阵和结果已保存。")

if __name__ == "__main__":
    main() 