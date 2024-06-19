import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import os
import warnings

def process_well_log_data(input_df, classifier_number:int, clip_length:int):
    """
    處理井測井數據，將數據分割成定長片段。

    Args:
        input_df (DataFrame): 包含原始測井數據的DataFrame。
        classifier_number (int): 標籤類別數。
        clip_length (int): 每段數據的長度。

    Returns:
        df_section (DataFrame): 處理後的數據段DataFrame。
    """
    # 初始化結果列表
    SP, N16, N64, GAMMA, ANS, WELL, Depth, ROCK_type = [], [], [], [], [], [], [], []
    class_dict = {2: 'class_2', 3: 'class_3', 4: 'class_4'}
    ans_class = class_dict[classifier_number]
    # 獲取唯一井號
    unique_well = input_df["number"].unique()

    # 遍歷每個井號的數據
    for i in tqdm(unique_well, desc="Processing well data"):
        temp = input_df[input_df["number"] == i]
        sp = temp["SP"].to_numpy()
        n16 = temp["N16"].to_numpy()
        n64 = temp["N64"].to_numpy()
        gamma = temp["GAMMA"].to_numpy()
        ans = temp[ans_class].to_numpy()
        depth = temp["depth"].to_numpy()
        rock_type = temp["type"].to_numpy()

        j = 0
        while j <= len(temp) - clip_length:
            interval = round(depth[j + clip_length - 1] - depth[j], 2)
            if not np.isnan(ans[j:j + clip_length]).any() and interval == round((clip_length - 1) * 0.1, 2):
                # 添加數據段到結果列表
                SP.append(sp[j:j + clip_length].tolist())
                N16.append(n16[j:j + clip_length].tolist())
                N64.append(n64[j:j + clip_length].tolist())
                GAMMA.append(gamma[j:j + clip_length].tolist())
                ANS.append(ans[j:j + clip_length].tolist())
                Depth.append(depth[j:j + clip_length].tolist())
                ROCK_type.append(rock_type[j:j + clip_length].tolist())
                WELL.append(i)
                j += clip_length  # 跳過clip_length長度
            else:
                j += 1  # 逐步遞增

    # 將結果轉換為DataFrame
    data_section = {
        'WELL': WELL,
        'Depth': Depth,
        'N16': N16,
        'N64': N64,
        'GAMMA': GAMMA,
        'SP': SP,
        'ANS': ANS,
        'type': ROCK_type
    }

    df_section = pd.DataFrame(data_section)
    return df_section

def split_test_df(df, test_number:str):
    """
    分割訓練和測試數據集。

    Args:
        df (DataFrame): 包含所有數據的DataFrame。
        test_number (str): 測試數據的井號。

    Returns:
        train_val_df (DataFrame): 訓練和驗證數據集。
        test_df (DataFrame): 測試數據集。
    """
    train_val_df = df[df['WELL'] != test_number].copy()
    test_df = df[df['WELL'] == test_number].copy()
    if len(test_df) == 0:
        print(f'沒有井號為 {test_number} 的資料')
        return None
    print(f'原始資料有{df.shape[0]}份')
    print(f'訓練驗證資料占{train_val_df.shape[0]}份')
    print(f'測試資料占{test_df.shape[0]}份')
    return train_val_df, test_df

def create_h5(train_val_df,
              test_df,
              class_type:str,
              for_index:int,
              output_path:str):
    """
    創建HDF5文件，包含訓練、驗證和測試數據集。

    Args:
        train_val_df (DataFrame): 訓練和驗證數據集。
        test_df (DataFrame): 測試數據集。
        class_type (str): 類別類型。
        for_index (int): 隨機狀態的迭代次數。
        output_path (str): HDF5文件的保存路徑。
    """
    for i in range(for_index):
        # 分割數據集並寫入HDF5文件
        df_dataset = train_val_df.sample(frac=1, random_state=i)

        split_point = math.ceil(len(df_dataset) * 0.8)
        train_df = df_dataset.iloc[:split_point]
        validation_df = df_dataset.iloc[split_point:]
        h5_name = f'class_{class_type}_TrainValTest_{i}.h5'
        file_path = os.path.join(output_path, h5_name)
        # 局部忽略 PerformanceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            with pd.HDFStore(file_path) as data:
                data['training'] = train_df
                data['validation'] = validation_df
                data['testing'] = test_df
