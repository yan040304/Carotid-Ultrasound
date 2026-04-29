import pandas as pd
import numpy as np

# ---- 修改为你的路径 ----
PRED_CSV = "results/diagnosis/longitudinal/evaluation_results.csv"
GT_CSV   = "results/diagnosis/longitudinal/ground_truth.csv"

# ---- 读取数据 ----
pred_df = pd.read_csv(PRED_CSV)
gt_df   = pd.read_csv(GT_CSV)

# 合并（基于 filename）
df = pred_df.merge(gt_df, on="filename")

y_true = df["gt_athero"].astype(int).to_numpy()
y_pred = df["athero"].astype(int).to_numpy()

# ---- 四类分类计数 ----
TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

# ---- 指标计算 ----
Accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-6)
Sensitivity = TP / (TP + FN + 1e-6)   # recall
Specificity = TN / (TN + FP + 1e-6)
Precision   = TP / (TP + FP + 1e-6)

# ---- 输出结果 ----
print("\n================ 评估结果 ================\n")
print(f"总样本数: {len(df)}")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}\n")

print(f"Accuracy    = {Accuracy:.4f}")
print(f"Sensitivity = {Sensitivity:.4f}")
print(f"Specificity = {Specificity:.4f}")
print(f"Precision   = {Precision:.4f}")

print("\n==========================================\n")

# ---- 保存到 CSV ----
out = {
    "Accuracy": Accuracy,
    "Sensitivity": Sensitivity,
    "Specificity": Specificity,
    "Precision": Precision,
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN
}
pd.DataFrame([out]).to_csv("results/diagnosis/longitudinal/method_metrics.csv", index=False)
print("结果已写入 results/diagnosis/longitudinal/method_metrics.csv")
