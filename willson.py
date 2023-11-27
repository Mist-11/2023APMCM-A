import statsmodels.api as sm
import numpy as np

# 假设数据 - 每次抽样的成功（正确预测）次数
successes = [209,210,209,210,210]  # 随机生成示例数据，您应该用实际的数据替换这里
n = 210  # 每次抽样的样本量

# 计算每次抽样的置信区间
for i, success in enumerate(successes, 1):
    ci_low, ci_upp = sm.stats.proportion_confint(success, n, alpha=0.01, method='wilson')
    print(f"抽样 {i}: 准确率 = {success/n:.6f}, 99% 置信区间 = ({ci_low:.6f}, {ci_upp:.6f})")


import statsmodels.api as sm
import numpy as np

# 示例数据：10次抽样的成功（正确预测）次数
successes = [209,210,209,210,210]  # 这里是随机生成的数据，您应该使用实际数据
n = 210  # 每次抽样的样本量

# 汇总所有抽样的结果
total_successes = np.sum(successes)
total_samples = n * len(successes)  # 总抽样次数

# 计算整体准确率
overall_accuracy = total_successes / total_samples

# 计算整体准确率的置信区间
ci_low, ci_upp = sm.stats.proportion_confint(total_successes, total_samples, alpha=0.01, method='wilson')

print(f"整体准确率: {overall_accuracy:.6f}")
print(f"整体准确率的99%置信区间: ({ci_low:.6f}, {ci_upp:.6f})")






###样本抽取
import pandas as pd

# 从CSV文件加载数据
data = pd.read_csv("C:\\Users\\x\\Desktop\\prediction_results.csv")

# 按'code'列排序
data_sorted = data.sort_values(by='code')

# 初始化空的DataFrame来存储最终的抽样结果
sampled_data = pd.DataFrame()

# 每1000个样本分为一组
for start in range(0, len(data_sorted), 1000):
    end = start + 1000
    group = data_sorted.iloc[start:end]

    # 如果组内样本少于10个，则取全部样本
    n_samples = min(len(group), 10)

    # 在每个组内抽取样本
    group_sample = group.sample(n=n_samples, random_state=1)
    sampled_data = sampled_data.append(group_sample)


sorted_data = sampled_data.sort_values(by='code')
sorted_data
sorted_data.to_csv("C:\\Users\\x\\Desktop\\222.csv", index=False)