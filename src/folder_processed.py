import tkinter as tk
from tkinter import filedialog
import os
import yaml

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
