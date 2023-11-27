import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./result/prediction_results.csv')
value_counts = df['Label'].value_counts()
total_counts = len(df['Label'])
# 计算占比
value_proportions = value_counts / total_counts * 100
# 打印结果
print("数量:\n", value_counts)
print("占比:\n", value_proportions)

sns.set(style="whitegrid")
# 分组并计算每组中0的数量
df['Index'] = df['Image Name'].str.extract('(\d+)')
df['Index'] = pd.to_numeric(df['Index'])
df.set_index('Index', inplace=True)
df.sort_index(inplace=True)
counts_per_1000 = df['Label'].groupby(df.index // 1000).apply(lambda x: (x == 0).sum())

# 使用Seaborn绘制直方图
plt.figure(figsize=(10, 6))
sns.barplot(x=counts_per_1000.index, y=counts_per_1000.values)
plt.title('Apple image ID number distribution histogram')
plt.xlabel('Group (Each group represents 1000 IDs)')
plt.ylabel('Number of Apple')
plt.savefig('./result/image/ID histogram.png')
plt.show()
