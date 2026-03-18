import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n--- {name} Evaluation ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}")
    
    return {
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }

def save_comparison_chart(results, path):
    models = [r['model'] for r in results]
    accuracy = [r['accuracy'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1 = [r['f1'] for r in results]
    roc_auc = [r['roc_auc'] for r in results]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    groups_data = [accuracy, precision, recall, f1, roc_auc]

    x = range(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'green', 'orange']
    
    num_models = len(models)
    
    for i, model_name in enumerate(models):
        model_scores = [g[i] for g in groups_data]
        bar_x = [pos + (i - (num_models - 1) / 2) * width for pos in x]
        bars = ax.bar(bar_x, model_scores, width, label=model_name, color=colors[i % len(colors)])
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Scores')
    ax.set_title('Model comparison — pit stop classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, facecolor='white', transparent=False)
    print(f"\nComparison chart saved to {path}")
