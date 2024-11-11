# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:02:02 2024

@author: lausa
"""

import pandas as pd
import matplotlib.pyplot as plt  # Plotting library


df = pd.read_csv("C:\\Users\\lausa\\OneDrive\\Desktop\\Thesis\\Project\\brenda_2024_1_WTs.csv")

print(df)

#print(df["sequence"].str.len())

plt.hist(df["sequence"].str.len(), bins=200)
plt.axis(xmax=1000, xmin=0)
# plt.hist(len(df.loc(df.sequence)))
# plt.show()

print(len(df.loc[df["sequence"].str.len() <= 1024]))

print(len(df.loc[df["sequence"].str.len() <= 1024])/len((df["sequence"].str.len()))*100)


# Exclude proteins over 1024 AA length

long = (df.sequence.str.len() > 1024) | (df.sequence.isna())

df_length = df.loc[~long]

print(df.loc[long])
print(df_length)

print(len(df_length.loc[df_length["sequence"].str.len() <= 1024]))
print(len(df_length.loc[df_length["sequence"].str.len() <= 1024])/len((df_length["sequence"].str.len()))*100)

print(max(df_length["sequence"].str.len()))

df_length.to_csv("C:\\Users\\lausa\\OneDrive\\Desktop\\Thesis\\Project\\brenda_2024_1_WTs_length.csv")