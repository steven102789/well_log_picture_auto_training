import os
import pandas as pd
import yaml
from scripts.run import model_selector
import src.folder_processed as fp
from src.well_logg_processing import process_well_log_data, split_test_df, create_h5

def main():
    #選擇模型
    experiments_path = None  # 初始化为 None
    model_dict = {'0':'CNN', '1':'U-Net'}
    select_model_number = input("請選擇要訓練的模型:\n0: CNN\n1: U-Net\n請輸入數字選擇模型: ").strip()

    if select_model_number not in model_dict:
        print("無效輸入，請輸入 0 或 1。")
        return

    print(f'已選擇{model_dict[select_model_number]}')

    if select_model_number == '0':  # CNN
        # 要做幾類別 ?
        classifier_number = select_classifier_number()
        # 是否要製作100m取樣?
        store_h5_folder_path = select_well_log_processing(classifier_number)  # 輸出存放h5的資料夾
        # 是否需要制作實驗資料夾 ?
        experiments_path = select_create_experiment_folder(input_dataset_folder=store_h5_folder_path,
                                                           classifier_number=classifier_number,
                                                           for_index=10)
    elif select_model_number == '1': #CNN
        # 要做幾類別 ?
        classifier_number = select_classifier_number()
        classifier_number += 1  # U-Net 多一個背景
        # 要做哪一個case
        case_dataset_folder = fp.select_directory('請選擇case文件夹')
        # 是否需要制作實驗資料夾 ?
        experiments_path = select_create_experiment_folder(input_dataset_folder=case_dataset_folder,
                                                           classifier_number=classifier_number,
                                                           for_index=10)


    # 確保 experiments_path 已經被賦值
    if experiments_path is None:
        raise ValueError("未能獲取資料夾位置。")

    # 獲取實驗資料夾中的子資料夾列表
    experiments_folder_list = [f for f in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, f))]

    # 檢查是否有子資料夾
    if not experiments_folder_list:
        print("沒有實驗數據")
        return
    print('開始訓練')
    # 遍歷實驗資料夾中的子資料夾
    for each_experiment_folder in experiments_folder_list:
        experiment_path = os.path.join(experiments_path, each_experiment_folder)
        config_path = os.path.join(experiment_path, 'config.yaml')
        # 確認 config.yaml 文件是否存在
        if os.path.exists(config_path):
            # 讀取 config.yaml 檔案
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                model_selector(select_model_number, config)
        else:
            print(f"{config_path} 不存在")
def select_classifier_number() -> int:
    classifier_number = int(input("要做幾類別(2/3/4):").strip())
    if classifier_number not in [2, 3, 4]:
        raise ValueError("只能输入2、3 或 4。")
    return  classifier_number
def select_create_experiment_folder(input_dataset_folder:str, classifier_number:int, for_index) ->str:
    select_create_experiment = input("是否需要制作實驗資料夾 ?: (y/n)").strip().lower()

    if select_create_experiment == 'y':
        select_experiments_dir = fp.select_directory('請選擇實驗文件夹', initial_dir='experiments')
        fp.create_experiment_folder(source_folder=input_dataset_folder,
                                 experiment_folder=select_experiments_dir,
                                 class_number=classifier_number,
                                 for_index=for_index)
        select_experiments_dir = os.path.join(select_experiments_dir , f'class_{classifier_number}')
        return select_experiments_dir
    elif select_create_experiment == 'n':
        return fp.select_directory('請選擇製作好的文件夹', initial_dir='experiments/')
    else:
        raise ValueError("只能输入 'y' 或 'n'。")
def select_well_log_processing(classifier_number:int) -> str:
    select_create_experiment = input("是否要製作wellLogging取樣100m ?: (y/n)").strip().lower()

    if select_create_experiment == 'y':
        select_source_csv_path = fp.select_file('请输入csv', 'create_well_logging_h5/input_data')
        source_csv = pd.read_csv(select_source_csv_path, index_col=False)
        # 取樣100m
        df_section = process_well_log_data(source_csv, classifier_number, 1000)
        select_test_number= str(input("請輸入測試井井號:").strip())
        train_val_df, test_df = split_test_df(df_section, select_test_number)
        temp_output_folder = fp.select_directory('請選擇dataset暫存位置', 'create_well_logging_h5/output_data/')
        print(f'已選擇{temp_output_folder}')
        fp.clear_directory(temp_output_folder)#清除先前文件
        create_h5(train_val_df,
                  test_df,
                  class_type=str(classifier_number),
                  for_index=10,
                  output_path=temp_output_folder)
        csv_name = os.path.join(temp_output_folder, f'class{classifier_number}_df_section.csv')
        df_section.to_csv(csv_name, index=False, encoding='utf-8-sig')
        return temp_output_folder

    elif select_create_experiment == 'n':
        temp_output_folder = fp.select_directory('請選擇dataset暫存位置', 'create_well_logging_h5/output_data/')
        return temp_output_folder
    else:
        raise ValueError("只能输入 'y' 或 'n'。")

if __name__ == "__main__":
    main()