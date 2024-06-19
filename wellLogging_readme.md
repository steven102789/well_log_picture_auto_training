# 此程式可將U-Net、CNN和Attention繁瑣的訓練過程自動化(wellLogging .ver)
1. 開啟cmd 進入python env(最後well_work為環境名稱)
```
conda activate well_work
```
2. cd 到資料夾位置(**注意!! 檔案位置不能有中文**)
```
cd 檔案位置
```
3. 輸入指令啟用程式碼
```
python main.py
```
-------
問題解釋:
1. 請選擇要訓練的模型:
    * 這裡選擇cnn 按下0
2. 要做幾類別(2/3/4)?
3. 是否需要取樣100m?
    * **需要的話:**
        * 需要的話請點選`create_well_logging_h5/input/`中的CSV文件
        * 如需測試取樣長度需到`main.py`中的`select_well_log_processing()`做修改
        * 輸入測試井號:(暫定冬山)`020853G1`
        * 製作好的h5檔電腦會詢問放在output資料夾何處，請在當中**新建**存放用的資料夾，資料夾名用英文
    * **不用的話選取存放h5的資料夾**
4. 是否需要建立實驗資料夾?
    * 實驗資料夾是用於管理與存放實驗結果的。
    * **需要的話:**
        * 請在`experiments/`建立需要的資料夾
        * 選擇剛建好的資料夾
        * 電腦會在當中新建`class_x/`資料夾並放入0~10的子資料夾<br>(如需調正超參數請到`src/folder_processed.py`中的`create_experiment_folder()`修改)
    * **不需要的話:(如因意外中斷需要重跑)**
        * 點選`experiments/建好的實驗資歷夾/class_x/`畫面要是0~10資料夾才行
5. 開始訓練模型