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

def clear_directory(directory_path: str):
    """
    清空指定目錄中的所有文件，但保留子目錄。

    Args:
        directory_path (str): 目標目錄路徑。

    Returns:
        None
    """
    # 确认目录存在
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return

    # 遍历目录中的所有文件和文件夹
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # 删除文件
        elif os.path.isdir(item_path):
            # 清空子目录中的文件（递归调用）
            clear_directory(item_path)

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