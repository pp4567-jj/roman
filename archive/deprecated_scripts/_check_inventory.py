import pandas as pd
df = pd.read_csv(r'd:\通过拉曼光谱预测物及其浓度\data\inventory.csv')
print(f'总行数: {len(df)}')
print(f'可读: {df["read_ok"].sum()}')
print(f'不可读: {(~df["read_ok"]).sum()}')
print()
print('格式分布:')
print(df['format_type'].value_counts())
print()
print('点数分布:')
print(df[df['read_ok']==True]['n_points'].value_counts())
print()
print('文件夹数:', df['folder_name'].nunique())
print()
print('各文件夹文件数:')
print(df.groupby('folder_name')['file_name'].count().to_string())
print()
print('异常文件:')
anomaly = df[df['read_ok']==False]
if len(anomaly) > 0:
    for _, r in anomaly.iterrows():
        print(f"  {r['relative_path']} -> {r['notes']}")
else:
    print('  无异常文件')
