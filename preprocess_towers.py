import pandas as pd

# read raw dataset
df = pd.read_csv("data/404.csv")

# rename longitude column
df = df.rename(columns={'long': 'lon'})

# keep only useful radio types (optional but good)
df = df[df['radio'].isin(['LTE','NR','UMTS','GSM'])]

# convert range from meters to km
df['range'] = df['range'] / 1000.0

# save cleaned file
df[['lat','lon','radio','range']].to_csv("data/towers_india.csv", index=False)

print("✅ towers_india.csv created successfully")