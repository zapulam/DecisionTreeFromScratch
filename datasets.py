"""
Written by Zachary Pulliam

Contains 3 dataset classes for the three example datasets which are tested on...
Other dataset classes can be added here.
"""

import os
import pandas as pd

"""Synthetic Datasets class for a single synthetic.csv file"""
class SyntheticDataset:
    def __init__(self, ROOT, frame, k):
        self.ROOT = ROOT
        self.name = 'Synthetic Dataset'
        self.train = None
        self.test = None
        self.frame = frame
        self.k = k
        self.bins = []
        self.load_data()

    def load_data(self):
        df = pd.read_csv(os.path.join(self.ROOT, 'synthetic-{x}.csv'.format(x=self.frame)), names=['A','B','Label'])

        exts = []
        for column in list(df)[0:-1]:
            min, max = df[column].min(), df[column].max()
            exts.append([min, max])

        for i, ext in enumerate(exts):
            vals = [ext[0]]
            for j in range(self.k-1):
                vals.append(round((ext[0] + (j+1)*((ext[1]-ext[0])/self.k)),3))
            vals.append(ext[1])
            self.bins.append(vals)
        labels = []
        for i, bin in enumerate(self.bins):
            labels.append([i for i in range(len(bin)-1)])
        
        df['A_bins'] = pd.cut(df['A'], bins=self.bins[0], labels=labels[0], include_lowest=True)
        df['B_bins'] = pd.cut(df['B'], bins=self.bins[1], labels=labels[1], include_lowest=True)

        self.train = df[['A_bins', 'B_bins', 'Label']].copy()
        self.test = df[['A_bins', 'B_bins', 'Label']].copy()

"""Synthetic Datasets class for a single synthetic.csv file seperated into folds for testing and training"""
class FoldedSyntheticDataset:
    def __init__(self, ROOT, frame, folds, k):
        self.ROOT = ROOT
        self.name = 'Folded Synthetic Dataset'
        self.trains = []
        self.tests = []
        self.frame = frame
        self.folds = folds
        self.k = k
        self.bins = []
        self.load_data()


    def load_data(self):
        df = pd.read_csv(os.path.join(self.ROOT, 'synthetic-{x}.csv'.format(x=self.frame)), names=['A','B','Label'])
        length = len(df.index)

        # creating bins
        exts = []
        for column in list(df)[0:-1]:
            min, max = df[column].min(), df[column].max()
            exts.append([min, max])

        # creating bin labels
        for i, ext in enumerate(exts):
            vals = [ext[0]]
            for j in range(self.k-1):
                vals.append(round((ext[0] + (j+1)*((ext[1]-ext[0])/self.k)),3))
            vals.append(ext[1])
            self.bins.append(vals)
        labels = []
        for i, bin in enumerate(self.bins):
            labels.append([i for i in range(len(bin)-1)])

        # binning columns
        df['A_bins'] = pd.cut(df['A'], bins=self.bins[0], labels=labels[0], include_lowest=True)
        df['B_bins'] = pd.cut(df['B'], bins=self.bins[1], labels=labels[1], include_lowest=True)

        # creating test folds
        for i in range(self.folds):
            rows = int(length/self.folds)
            test = df.iloc[i*rows:(i+1)*rows]
            df.drop(df.index[i*rows:(i+1)*rows])
            
            df = df[['A_bins', 'B_bins', 'Label']].copy()
            test = test[['A_bins', 'B_bins', 'Label']].copy()

            # append fold to list of folds
            self.trains.append(df)
            self.tests.append(test)


"""Pokemon!!!"""
class PokemonDataset:
    def __init__(self, ROOT, k):
        self.ROOT = ROOT
        self.name = 'Pokemon Dataset'
        self.train = None
        self.test = None
        self.k = k
        self.bins = []
        self.load_data(self.ROOT)

    def load_data(self, ROOT):
        df_data = pd.read_csv(ROOT + '/pokemonStats.csv')
        df_data['Type_1'] = 0
        df_data['Type_2'] = 0

        # creating single columns for 'Type 1' and 'Type 2'
        for i, row in df_data.iterrows():
            for j in range(8,26):
                if row.iloc[j] == 1:
                    df_data.iloc[i,44] = j-7
            for j in range(26,44):
                if row.iloc[j] == 1:
                    df_data.iloc[i,45] = j-25
        
        # dropping extra type columns
        cols = [i for i in range(8,44)]
        df_data.drop(df_data.columns[cols], axis=1, inplace=True)

        # creating bins
        exts = []
        for column in list(df_data)[0:-3]:
            min, max = df_data[column].min(), df_data[column].max()
            exts.append([min, max])

        # creating labels for bins
        for i, ext in enumerate(exts):
            vals = [ext[0]]
            for j in range(self.k-1):
                vals.append(round((ext[0] + (j+1)*((ext[1]-ext[0])/self.k)),3))
            vals.append(ext[1])
            self.bins.append(vals)
        labels = []
        for i, bin in enumerate(self.bins):
            labels.append([i for i in range(len(bin)-1)])

        # binning all columns
        df_data['Total_bins'] = pd.cut(df_data['Total'], bins=self.bins[0], labels=labels[0], include_lowest=True)
        df_data['HP_bins'] = pd.cut(df_data['HP'], bins=self.bins[1], labels=labels[1], include_lowest=True)
        df_data['Attack_bins'] = pd.cut(df_data['Attack'], bins=self.bins[2], labels=labels[2], include_lowest=True)
        df_data['Defense_bins'] = pd.cut(df_data['Defense'], bins=self.bins[3], labels=labels[3], include_lowest=True)
        df_data['SpAttack_bins'] = pd.cut(df_data['Sp. Atk'], bins=self.bins[4], labels=labels[4], include_lowest=True)
        df_data['SpDef_bins'] = pd.cut(df_data['Sp. Def'], bins=self.bins[5], labels=labels[5], include_lowest=True)
        df_data['Speed_bins'] = pd.cut(df_data['Speed'], bins=self.bins[6], labels=labels[6], include_lowest=True)

        df_binned_data = df_data[['Total_bins', 'HP_bins', 'Attack_bins', 'Defense_bins', 'SpAttack_bins', 'SpDef_bins', 'Speed_bins', 'Generation', 'Type_1', 'Type_2']].copy()
        df_label = pd.read_csv(ROOT + '/pokemonLegendary.csv')

        self.train = pd.concat([df_binned_data, df_label], axis=1)
        self.test = pd.concat([df_binned_data, df_label], axis=1)
