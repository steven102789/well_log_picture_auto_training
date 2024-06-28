import json
import tkinter as tk
from tkinter import filedialog
import os
import yaml
from src.well_logg_processing import process_well_log_data, split_test_df, create_h5
import pandas as pd

def create_experiment_folder(source_folder: str,
                             experiment_folder: str,
                             class_number: int,
                             for_index: int):
    """
    創建實驗資料夾，並生成相應的 config.yaml 文件。

    Args:
        source_folder (str): 資料來源資料夾。
        experiment_folder (str): 實驗資料夾。
        class_number (int): 類別數量。
        for_index (int): 循環次數，用於設定不同的 random_state。

    Returns:
        None
    """
    for index in range(for_index):
        # 定义要写入 config.yaml 的内容
        config_data = {
            'input_location': f'{source_folder}',
            'input_shape': 1000,
            'kernel_size': 3,
            'n_classes': class_number,
            'batch_size': 32,
            'learning_rate': 0.00001,
            'random_state': index,
            'output_location': f'{experiment_folder}class_{class_number}/{index}/output/',
            'excel_output_location': f'{experiment_folder}class_{class_number}/',
        }

        # 定义资料夹名称和路径
        folder_name = str(index)
        folder_path = os.path.join(experiment_folder, f'class_{class_number}', folder_name)

        # 检查资料夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 创建 output 文件夹
        output_folder_path = os.path.join(folder_path, 'output')
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # 定义 config.yaml 的路径
        config_file_path = os.path.join(folder_path, 'config.yaml')

        # 写入 config.yaml 文件
        with open(config_file_path, 'w') as config_file:
            yaml.dump(config_data, config_file)
    print('實驗資料夾製作完成!!')

def select_directory(title: str, initial_dir: str = None) -> str:
    """
    打開文件對話框以選擇目錄。

    Args:
        title (str): 對話框標題。
        initial_dir (str, optional): 初始目錄。默認為 None。

    Returns:
        str: 選擇的目錄路徑。
    """
    root = tk.Tk()
    root.withdraw()  # 隱藏主窗口
    root.attributes('-topmost', True)  # 使主窗口保持在最前面
    root.update()  # 更新窗口以應用屬性

    directory_path = filedialog.askdirectory(title=title, initialdir=initial_dir)  # 打開文件對話框選擇目錄
    root.destroy()  # 銷毀主窗口

    if directory_path and not directory_path.endswith(os.sep):
        directory_path += os.sep
    return directory_path

def select_file(title: str, initial_dir: str = None) -> str:
    """
    打開文件對話框以選擇文件。

    Args:
        title (str): 對話框標題。
        initial_dir (str, optional): 初始目錄。默認為 None。

    Returns:
        str: 選擇的文件路徑。
    """
    root = tk.Tk()
    root.withdraw()  # 隱藏主窗口
    root.attributes('-topmost', True)  # 使主窗口保持在最前面
    root.update()  # 更新窗口以應用屬性

    file_path = filedialog.askopenfilename(title=title, initialdir=initial_dir)  # 打開文件對話框選擇文件
    root.destroy()  # 銷毀主窗口

    return file_path

def select_classifier_number() -> int:
    """
        提示用戶選擇分類數量。

        Returns:
            int: 選擇的分類數量 (2, 3 或 4)。

        Raises:
            ValueError: 如果輸入的不是 2、3 或 4。
        """
    classifier_number = int(input("要做幾類別(2/3/4):").strip())
    if classifier_number not in [2, 3, 4]:
        raise ValueError("只能输入2、3 或 4。")
    return  classifier_number

def select_create_experiment_folder(input_dataset_folder:str, classifier_number:int, for_index) ->str:
    """
    根據用戶選擇創建或選擇實驗資料夾。

    Args:
        input_dataset_folder (str): 輸入數據集文件夾的路徑。
        classifier_number (int): 分類器的類別數量 (2, 3 或 4)。
        for_index (int): 創建實驗資料夾的索引範圍。

    Returns:
        str: 選擇或創建的實驗資料夾路徑。

    Raises:
        ValueError: 當輸入不是 'y' 或 'n' 時引發錯誤。
    """
    select_create_experiment = input("是否需要制作實驗資料夾 ?: (y/n)").strip().lower()

    if select_create_experiment == 'y':
        select_experiments_dir = select_directory('請選擇實驗文件夹', initial_dir='experiments')
        create_experiment_folder(source_folder=input_dataset_folder,
                                 experiment_folder=select_experiments_dir,
                                 class_number=classifier_number,
                                 for_index=for_index)
        select_experiments_dir = os.path.join(select_experiments_dir , f'class_{classifier_number}')
        return select_experiments_dir
    elif select_create_experiment == 'n':
        return select_directory('請選擇製作好的文件夹', initial_dir='experiments/')
    else:
        raise ValueError("只能输入 'y' 或 'n'。")

def select_well_log_processing(classifier_number:int) -> str:
    """
        根據用戶選擇進行well log數據處理，並返回暫存位置。

        Args:
            classifier_number (int): 分類器的類別數量 (2, 3 或 4)。

        Returns:
            str: 選擇或創建的暫存資料夾路徑。

        Raises:
            ValueError: 當輸入不是 'y' 或 'n' 時引發錯誤。
        """
    select_create_experiment = input("是否要製作wellLogging取樣100m ?: (y/n)").strip().lower()

    if select_create_experiment == 'y':
        select_source_csv_path = select_file('请输入csv', 'create_well_logging_h5/input_data')
        source_csv = pd.read_csv(select_source_csv_path, index_col=False)
        # 取樣100m
        df_section = process_well_log_data(source_csv, classifier_number, 1000)
        select_test_number= str(input("請輸入測試井井號:").strip())
        train_val_df, test_df = split_test_df(df_section, select_test_number)
        temp_output_folder = select_directory('請選擇dataset暫存位置', 'create_well_logging_h5/output_data/')
        print(f'已選擇{temp_output_folder}')
        clear_directory(temp_output_folder)#清除先前文件
        create_h5(train_val_df,
                  test_df,
                  class_type=str(classifier_number),
                  for_index=10,
                  output_path=temp_output_folder)
        csv_name = os.path.join(temp_output_folder, f'class{classifier_number}_df_section.csv')
        df_section.to_csv(csv_name, index=False, encoding='utf-8-sig')
        return temp_output_folder

    elif select_create_experiment == 'n':
        temp_output_folder = select_directory('請選擇dataset暫存位置', 'create_well_logging_h5/output_data/')
        return temp_output_folder
    else:
        raise ValueError("只能输入 'y' 或 'n'。")

def clear_directory(directory_path: str):
    """
    清除指定目录中的所有文件，但保留目录结构。

    Parameters:
    - directory_path (str): 要清除的目录路径。
    """
    # 确保路径存在且是一个目录
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        raise ValueError(f"路径不存在或不是目录: {directory_path}")

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"无法删除文件 {file_path}: {e}")

def create_training_log(experiments_path: str, sub_folder_list: list):
    """
    在指定的实验路径下创建训练日志文件 `training_log.json`。

    Parameters:
    - experiments_path (str): 实验文件夹的路径。
    - sub_folder_list (list): 含有所有子資料夾名稱的列表

    """
    log_path = os.path.join(experiments_path, 'training_log.json')

    if not os.path.exists(log_path):
        # 如果日志文件不存在，则创建一个新的日志文件
        log_content = {
            'logs': sub_folder_list
        }
        with open(log_path, 'w') as log_file:
            json.dump(log_content, log_file, indent=4)

        print(f"已创建新的训练日志文件: {log_path}")
    else:
        print(f"训练日志文件已存在: {log_path}")

def read_training_log_from_directory(directory_path: str) -> list:
    """
    从指定的目录中读取 `training_log.json` 文件，并将 `logs` 字段内容加载为一个列表对象。

    Parameters:
    - directory_path (str): 包含 `training_log.json` 文件的目录路径。

    Returns:
    - list: `logs` 字段内容的列表。
    """
    log_path = os.path.join(directory_path, 'training_log.json')

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"未找到日志文件: {log_path}")

    with open(log_path, 'r') as log_file:
        log_content = json.load(log_file)

    logs = log_content.get('logs', [])

    if not isinstance(logs, list):
        raise ValueError(f"日志文件中的 `logs` 字段内容不是列表: {log_path}")

    return logs

def compare_lists(sub_folder_list: list, training_log_list: list) -> list:
    """
    比较两个列表，如果它们完全相同则返回 `sub_folder_list`，否则返回 `training_log_list`。

    Parameters:
    - sub_folder_list (list): 子文件夹名称的列表。
    - training_log_list (list): 从日志文件中读取的列表。

    Returns:
    - list: 如果两个列表相同则返回 `sub_folder_list`，否则返回 `training_log_list`。
    """
    if sub_folder_list == training_log_list:
        return sub_folder_list
    else:
        return training_log_list

def update_training_log(log_path, completed_folder):
    """
    更新训练日志文件，将已完成的文件夹从日志中移除。

    Parameters:
    - log_path (str): 训练日志文件的路径。
    - completed_folder (str): 已完成的实验文件夹名称。
    """
    log_path = os.path.join(log_path, 'training_log.json')

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    with open(log_path, 'r') as log_file:
        log_content = json.load(log_file)

    if 'logs' in log_content:
        log_content['logs'].remove(completed_folder)

        with open(log_path, 'w') as log_file:
            json.dump(log_content, log_file, indent=4)

        print(f"更新训练日志文件: {log_path}")
    else:
        raise KeyError(f"日志文件中不存在 'logs' 键: {log_path}")