import pandas as pd
import numpy as np
from osgeo import gdal
import re
from functools import reduce
import pingouin as pg
from statsmodels.stats.inter_rater import fleiss_kappa

JUDGES = ['1', '2', '3']

def parse_df(area=True):

    def read(kind, judge):
        return pd.read_excel(f'labelled_data/building_images_{kind}_{judge}.xlsx', thousands='.')

    dfs = []
    for judge_num, judge in enumerate(JUDGES):
        df = read('individual', judge)
        df = df.rename(columns={
            'DAMAGE [0-100%]': f'DAMAGE_{judge}',
            'BUILDING_CLASS (G/M/B)': f'BUILDING_CLASS_{judge}'
        })
        if judge_num == 0:
            df = df[['ID', f'BUILDING_CLASS_{judge}', f'DAMAGE_{judge}', 'LONGITUDE', 'LATITUDE']]
        else:
            df = df[['ID', f'BUILDING_CLASS_{judge}', f'DAMAGE_{judge}']]
        df.loc[df[f'BUILDING_CLASS_{judge}'] == 'H', f'BUILDING_CLASS_{judge}'] = 'G'
        
        if area:
            cur_id = 10000 # higher than individual IDs
            df_area = read('area', judge)
            for _, damage_area in df_area.iterrows():
                damages = []
                for damage_observations in damage_area['DAMAGE'].split(','):
                    n, building_class, damage = re.findall(r'([0-9]+)([A-Z])([0-9]+)', damage_observations)[0]
                    n = int(n)
                    damage = int(damage) / 100
                    for _ in range(n):
                        damages.append((damage, building_class))
                damages = sorted(damages, key=lambda x: x[0])
                for damage, building_class in damages:
                    observation = {
                        'ID': cur_id,
                        f'DAMAGE_{judge}': damage,
                        f'BUILDING_CLASS_{judge}': building_class
                    }
                    if judge_num == 0:
                        observation.update({
                            'LONGITUDE': damage_area['LONGITUDE'],
                            'LATITUDE': damage_area['LATITUDE'],
                        })
                    df = pd.concat([df, pd.DataFrame([observation])], ignore_index=True)
                    cur_id += 1

        dfs.append(df)

    assert len(dfs[0]) == len(dfs[1]) == len(dfs[2])
    df = reduce(lambda left, right: pd.merge(left, right, on='ID'), dfs)
    for judge in JUDGES:
        if df[f'DAMAGE_{judge}'].dtype == '|O':
            df = df[df[f'DAMAGE_{judge}'].str.isnumeric().isnull()]

    len_before_drop = len(df)
    df = df.dropna(how='any')
    print(f'Dropped {len_before_drop - len(df)} rows with NA')
    return df


class GetWindSpeed:
    def __init__(self):
        self.ds = gdal.Open('max_wind_field.tif')
        self.gt = self.ds.GetGeoTransform()
        self.rb = self.ds.GetRasterBand(1)

    def get_px_py(self, mx, my):
        px = int((mx - self.gt[0]) / self.gt[1]) #x pixel
        py = int((my - self.gt[3]) / self.gt[5]) #y pixel
        return px, py

    def __call__(self, lon, lat):
        px, py = self.get_px_py(lon, lat)
        return self.rb.ReadAsArray(px, py, 1, 1)[0][0] * 3.6



def calculate_fleiss_kappa(df):
    table = df[[f'BUILDING_CLASS_{judge}' for judge in JUDGES]].to_numpy().astype('U')
    fleiss_table = np.zeros_like(table, dtype=np.int32)
    fleiss_table[:, 0] = (table == 'G').sum(axis=1)
    fleiss_table[:, 1] = (table == 'M').sum(axis=1)
    fleiss_table[:, 2] = (table == 'B').sum(axis=1)
    assert (fleiss_table.sum(1) == 3).all()
    return fleiss_kappa(fleiss_table)

def calculate_icc(df):
    icc_data = []
    for i, row in df.iterrows():
        for judge in JUDGES:
            icc_data.append((row['ID'], judge, row[f'BUILDING_CLASS_{judge}'], row[f'DAMAGE_{judge}']))

    icc_df = pd.DataFrame(icc_data, columns=['ID', 'judge', 'class', 'damage'])

    assert (icc_df['damage'] <= 1).all()
    assert (icc_df['damage'] >= 0).all()
    assert (icc_df['class'].isin(['G', 'M', 'B'])).all()

    icc = pg.intraclass_corr(data=icc_df, targets='ID', raters='judge', ratings='damage', nan_policy='raise').round(3)
    return icc

def assign_median_building_class(df):
    building_classes = df[[f'BUILDING_CLASS_{judge}' for judge in JUDGES]]
    building_classes[building_classes == 'G'] = 0
    building_classes[building_classes == 'M'] = 1
    building_classes[building_classes == 'B'] = 2
    building_class = building_classes.median(axis=1).astype(np.int32).astype('U')
    building_class[building_class == "0"] = 'G'
    building_class[building_class == "1"] = 'M'
    building_class[building_class == "2"] = 'B'
    df['BUILDING_CLASS'] = building_class
    return df

def assign_damage(df, mode):
    damages = df[[f'DAMAGE_{judge}' for judge in JUDGES]]
    if mode == 'median':
        damage = damages.median(axis=1)
    elif mode == 'mean':
        damage = damages.mean(axis=1)
    else:
        raise ValueError
    df['DAMAGE'] = damage
    return df

df = parse_df()

fleiss_kappa = calculate_fleiss_kappa(df)
print('fleiss kappa', fleiss_kappa)
icc = calculate_icc(df)
print('intraclass correlation', icc)

df = assign_median_building_class(df)
df = assign_damage(df, mode='median')

get_wind_speed = GetWindSpeed()
df['WIND_SPEED'] = df.apply(lambda x: get_wind_speed(x.LONGITUDE, x.LATITUDE), axis=1)

rename = {
    'G': 'good',
    'M': 'medium',
    'B': 'bad'
}

for building_class, building_class_data in df.groupby('BUILDING_CLASS'):
    observations = building_class_data[['WIND_SPEED', 'DAMAGE']].rename(
        columns={'WIND_SPEED': 'x', 'DAMAGE': 'y'}
    )
    observations.to_csv(f'observations_{rename[building_class]}.csv', index=False)