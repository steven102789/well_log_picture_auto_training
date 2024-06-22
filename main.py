from scripts.run import model_selector
from src.folder_processed import *
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
        case_dataset_folder = select_directory('請選擇case文件夹')
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


if __name__ == "__main__":
    main()