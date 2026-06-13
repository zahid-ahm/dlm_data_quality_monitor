def profile(batch):
    return {'rows': len(batch),
            'null_rate': batch['region'].isna().mean(),
            'mean_amount': batch['amount'].mean()}
