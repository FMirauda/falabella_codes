import pandas as pd
from google.cloud import bigquery

def cusdis(df, n_rows=2):
	"""Custom display"""
	display(df.head(n_rows))
	print(df.shape)

def show(df, rows=False, columns=False, cells=False):
	params = []

	if rows:
		params = params + ['max_rows', None]

	if columns:
		params = params + ['max_columns', None]

	if cells:
		params = params + ['max_cellwidth', None]

	with pd.option_context(*params):
		display(df)

def read_bq(query, return_df = True):
    client = bigquery.Client()
    job = client.query(query)
    if return_df:
        return job.to_dataframe()

def dt_to_str(dt):
    return str(dt)[:10]
