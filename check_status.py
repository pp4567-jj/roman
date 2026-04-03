import pandas as pd
for tag in ['raw','p1','p2','p3','p4']:
    path = f'data/models/cv_results_summary_{tag}.csv'
    df = pd.read_csv(path)
    models = sorted(df['Model'].unique())
    print(f'{tag}: {len(df)} rows, models={models}')
