import pandas as pd


def pivot_result_by_features(path):
    df = pd.read_csv(path)
    aggregation_functions = {'Macro F1': 'mean'}
    grouped = df.groupby(['Created features', 'model']).aggregate(aggregation_functions)
    return grouped


if __name__ == '__main__':
    path = 'output/Greek/result.csv'
    pivot_result_by_features(path)
