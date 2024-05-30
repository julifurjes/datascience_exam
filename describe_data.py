import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory
# os.chdir('NDA-group/')
# Set working directory
# folder_path = '/path/to/your/folder'
# file_name = 'survey.csv'
# file_path = os.path.join(folder_path, file_name)
# df = pd.read_csv(file_path)
df = pd.read_csv('survey.csv')
print(df)

#### DESCRIBE DATA ####

# Describe the amount of people who sought treatment vs not
value_counts = df['treatment'].value_counts()
print(value_counts)

plt.figure(figsize=(8, 4))
value_counts.plot(kind='bar', color='purple')
plt.title('Bar Chart of Treatment Seeking')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.savefig('plots/data_desc/treatment_dist.png')
plt.clf()

# Describe country ratio
value_counts = df['Country'].value_counts()
print(value_counts)

# Keep only counts >= 10 and sum others under 'Other'
filtered_counts = value_counts[value_counts >= 10]
other_count = value_counts[value_counts < 10].sum()
if other_count > 0:
    filtered_counts['Other'] = other_count

plt.figure(figsize=(10, 10))
filtered_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, pctdistance=0.65, textprops={'size': 'smaller'})
plt.title('Country Distribution')
plt.ylabel('')
plt.savefig('plots/data_desc/country_dist.png')
plt.clf()

# Make sure to only keep ages that are realistic
df.drop(df[df['Age'] < 0].index, inplace=True)
df.drop(df[df['Age'] > 100].index, inplace=True)

# Visualise age distribution (with the filtered age)
plt.figure(figsize=(12, 8))
sns.distplot(df["Age"], color='purple', bins=24)
plt.title("Distribution and Density by Age")
plt.xlabel("Age")
plt.savefig('plots/data_desc/age_plot.png')
plt.clf()

#### DATA PREP ####

# Handling NAs
df['work_interfere'] = df['work_interfere'].fillna('Don\'t know')
print(df['work_interfere'].unique())

df['self_employed'] = df['self_employed'].fillna('No')  # Set NAs as 'No'
print(df['self_employed'].unique())

df = df.drop('state', axis=1)  # We dont care about the state and there is a lot of NAs there
df = df.drop('Timestamp', axis=1)  # Remove timestamp

print(df.isnull().sum())  # Count NAs

# Label encoding
from sklearn.preprocessing import LabelEncoder
object_cols = ['Gender', 'self_employed', 'family_history', 'treatment',
               'work_interfere', 'no_employees', 'remote_work', 'tech_company',
               'benefits', 'care_options', 'wellness_program', 'seek_help',
               'anonymity', 'leave', 'mental_health_consequence',
               'phys_health_consequence', 'coworkers', 'supervisor',
               'mental_health_interview', 'phys_health_interview',
               'mental_vs_physical', 'obs_consequence']
label_encoder = LabelEncoder()
for col in object_cols:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])

#### PLOTS ####

# Compute the correlation matrix
corr_df = df.drop(['Country', 'comments'], axis=1)

corr = corr_df.corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap='Purples', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.savefig('plots/data_desc/correlation_matrix.png')