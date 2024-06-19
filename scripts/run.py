import cv2
import glob
import gc
import os
import tqdm
import pandas as pd
import numpy as np
from model.CNNModel import CNNModel
from model.U_NetModel import UNet, weighted_loss
from src.DataProcessor import DataProcessor
from src.draw_plot import draw_train_history_plot, plot_confusion_matrix, lithology_plot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

def model_selector(select_number:str, config):
    if select_number == '0':
        train_cnn_model(config)
    elif select_number == '1':
        train_u_net_model(config)
    else:
        print("无效输入，请输入 0 或 1。")

def train_cnn_model(config):
    #class dictionary
    target_names = {2:["mud", "gravel"],
                    3:['mud', 'sand', 'gravel'],
                    4:['mud', 'fine sand', 'coarse sand', 'gravel']}

    # 輸入參數
    father_input_location = config['input_location']
    input_shape = config['input_shape']
    kernel_size = config['kernel_size']
    n_classes = config['n_classes']
    learning_rate = config['learning_rate']
    random_state = config['random_state']
    output_location = config['output_location']
    excel_output_location = config['excel_output_location']

    input_location = os.path.join(father_input_location, f'class_{n_classes}_TrainValTest_{random_state}.h5')

    try:
        # 尝试打开并读取HDF5文件中的数据集
        with pd.HDFStore(input_location) as data:
            train_dataset = data['training']
            validation_dataset = data['validation']
            test_dataset = data['testing']
    except KeyError as e:
        # 捕获KeyError异常并显示自定义错误消息
        print(f"KeyError: {e}. 文件位置錯誤")

    print(f'訓練資料筆數{train_dataset.shape[0]}')
    print(f'驗證資料筆數{validation_dataset.shape[0]}')
    print(f'驗證資料筆數{test_dataset.shape[0]}')

    ################################################################
    # 將資料轉為CNN input
    train_data_processor = DataProcessor(train_dataset, input_shape)
    val_data_processor = DataProcessor(validation_dataset, input_shape)
    test_data_processor = DataProcessor(test_dataset, input_shape)

    # processing training data
    train_16 = train_data_processor.preprocess_data("N16")
    train_64 = train_data_processor.preprocess_data("N64")
    train_gamma = train_data_processor.preprocess_data("GAMMA")
    train_sp = train_data_processor.preprocess_data("SP")
    x_train = [train_16, train_64, train_gamma, train_sp]

    # processing validation data
    val_16 = val_data_processor.preprocess_data("N16")
    val_64 = val_data_processor.preprocess_data("N64")
    val_gamma = val_data_processor.preprocess_data("GAMMA")
    val_sp = val_data_processor.preprocess_data("SP")
    x_val = [val_16, val_64, val_gamma, val_sp]

    # processing testing data
    test_16 = test_data_processor.preprocess_data("N16")
    test_64 = test_data_processor.preprocess_data("N64")
    test_gamma = test_data_processor.preprocess_data("GAMMA")
    test_sp = test_data_processor.preprocess_data("SP")
    x_test = [test_16, test_64, test_gamma, test_sp]

    # Convert validation observations to one hot vectors
    y_train_one_hot = to_categorical( np.array([x for x in train_dataset["ANS"]]), num_classes=n_classes)
    y_val_one_hot = to_categorical( np.array([x for x in validation_dataset["ANS"]]), num_classes=n_classes)
    y_test_one_hot = to_categorical(np.array([x for x in test_dataset["ANS"]]), num_classes=n_classes)
    ################################################################
    # 匯入模型
    model_instance =  CNNModel(input_shape=input_shape,
                               n_classes=n_classes,
                               kernel_size=kernel_size)
    merged = model_instance.build_model()

    # 模型訓練用的參數
    learning_rate_function = ReduceLROnPlateau(monitor='val_loss',
                                               patience=20,
                                               verbose=1,
                                               factor=0.06,
                                               min_lr=learning_rate)
    early_stop = EarlyStopping(monitor='val_loss',
                               mode='min',
                               patience=50,
                               verbose=1)
    model_check = ModelCheckpoint(output_location+'hydro_geo.h5',
                                  monitor='val_loss',
                                  mode='min',
                                  verbose=1,
                                  save_best_only=True)
    # 訓練模型
    merged.compile( optimizer=Adam(learning_rate= 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                    loss = 'categorical_crossentropy', metrics=['accuracy'])
    train_history=merged.fit(x= x_train,
                             y=y_train_one_hot,
                             epochs=300,
                             batch_size=1024,
                             callbacks=[learning_rate_function,early_stop, model_check],
                             validation_data=(x_val , y_val_one_hot),
                             verbose=2)
    draw_train_history_plot(train_history= train_history,
                            picture_name='loss&accuracy',
                            location=output_location)
    try:
        merged.load_weights(output_location+'hydro_geo.h5')
        print("載入模型成功!繼續訓練模型")
    except:
        print("載入模型失敗!開始訓練一個新模型")
    ################################################################
    # training dataset analysis
    y_train_prediction = merged.predict(x_train)
    # Convert predictions classes to one hot vectors
    y_train_prediction_classes = np.argmax(y_train_prediction, axis = -1)
    # Convert validation observations to one hot vectors
    y_train_true = np.argmax(y_train_one_hot,axis = -1)
    # compute the confusion matrix
    train_cm = confusion_matrix(y_train_true.ravel(),
                                y_train_prediction_classes.ravel())

    plot_confusion_matrix('train_cm',
                          location=output_location,
                          cm= train_cm,
                          classes=list(range(0,n_classes)))
    training_report = classification_report(y_train_true.ravel(),
                                            y_train_prediction_classes.ravel(),
                                            digits=4,
                                            target_names=target_names[n_classes],
                                            output_dict=True)
    ################################################################
    # validation dataset analysis
    y_validation_prediction = merged.predict(x_val)
    # Convert predictions classes to one hot vectors
    y_validation_prediction_classes = np.argmax(y_validation_prediction,axis = -1)
    # Convert validation observations to one hot vectors
    y_validation_true = np.argmax(y_val_one_hot,axis = -1)
    # compute the confusion matrix
    validation_cm = confusion_matrix(y_validation_true.ravel(),
                                     y_validation_prediction_classes.ravel())

    plot_confusion_matrix('validation_cm',
                          location=output_location,
                          cm= validation_cm,
                          classes=list(range(0,n_classes)))
    validation_report = classification_report(y_validation_true.ravel(),
                                              y_validation_prediction_classes.ravel(),
                                              digits=4,
                                              target_names=target_names[n_classes],
                                              output_dict=True)
    ################################################################
    # testing dataset analysis
    y_test_prediction = merged.predict(x_test)
    # Convert predictions classes to one hot vectors
    y_test_prediction_classes = np.argmax(y_test_prediction, axis = -1)
    # Convert validation observations to one hot vectors
    y_test_true = np.argmax(y_test_one_hot,axis = -1)
    # compute the confusion matrix
    test_cm = confusion_matrix(y_test_true.ravel(),
                               y_test_prediction_classes.ravel())

    plot_confusion_matrix('test_cm',
                          location=output_location,
                          cm= test_cm,
                          classes=list(range(0,n_classes)))
    testing_report = classification_report(y_test_true.ravel(),
                                           y_test_prediction_classes.ravel(),
                                           digits=4,
                                           target_names=target_names[n_classes],
                                           output_dict=True)
    #draw the lithology picture
    test_dataset["Prediction"] = y_test_prediction_classes.tolist()
    for i in range(len(test_dataset)):
        lithology_plot(dfRow=test_dataset[i:i + 1],
                       classifier_number=str(n_classes),
                       picture_name='lithology',
                       location=output_location)
    ################################################################
    # 製作excel
    train_df = pd.DataFrame(training_report).transpose()
    val_df = pd.DataFrame(validation_report).transpose()
    test_df = pd.DataFrame(testing_report).transpose()

    train_row_data, val_row_data, test_row_data = {}, {}, {}
    for col in val_df.columns:
        val_row_data[col] = 'validation'
        train_row_data[col] = 'train'
        test_row_data[col] = 'test'
    top_val_row = pd.DataFrame(val_row_data, index=[0])
    top_train_row = pd.DataFrame(train_row_data, index=[0])
    top_test_row = pd.DataFrame(test_row_data, index=[0])

    train_df = pd.concat([top_train_row, train_df], ignore_index=False)
    val_df = pd.concat([top_val_row, val_df], ignore_index=False, axis=0)
    test_df = pd.concat([top_test_row, test_df], ignore_index=False, axis=0)
    result_df = pd.concat([train_df, val_df, test_df], ignore_index=False, axis=0)

    # 检查文件是否存在
    excel_file_path = f'{excel_output_location}result.xlsx'
    if os.path.isfile(excel_file_path):
        # 如果文件已存在，在現有 Excel 檔案的指定工作表中寫入 DataFrame
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
            result_df.to_excel(writer, index=True, sheet_name=f'random_{random_state}')
            print('已寫入文件')
    else:
        # 如果文件不存在，創建新的 Excel 文件並寫入 DataFrame
        result_df.to_excel(excel_file_path, index=True, sheet_name=f'random_{random_state}')
        print('已寫入文件')
    print('訓練完成!!')

def train_u_net_model(config):
    target_names = {3: ['background', 'mud', 'gravel'],
                    4: ['background', 'mud', 'sand', 'gravel'],
                    5: ['background', 'mud', 'fine sand', 'coarse sand', 'gravel']}
    # 輸入參數
    input_location = config['input_location']
    batch_size = config['batch_size']
    n_classes = config['n_classes']
    learning_rate = config['learning_rate']
    random_state = config['random_state']
    output_location = config['output_location']
    excel_output_location = config['excel_output_location']

    # 調整圖片大小，如果有需要
    SIZE_X = 256
    SIZE_Y = 256
    # 捕捉訓練圖片信息，以列表形式保存
    train_images,train_masks = [], []
    for img_path in tqdm.tqdm(glob.glob(input_location + 'image/*.png'), desc="正在載入照片"):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)

    # 捕捉遮罩/標籤信息，以列表形式保存
    for mask_path in tqdm.tqdm(glob.glob(input_location + f'{n_classes}_label/*.png'), desc="正在載入標籤"):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        train_masks.append(mask)

    # 將列表轉換為陣列，以供機器學習處理
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)
    # 顯示詳細資料
    print("train_val image shape: " + str(train_images.shape))
    print("train_val mask shape: " + str(train_masks.shape))
    ###############################################
    # 捕捉測試圖片信息，以列表形式保存
    test_images, test_masks = [], []
    for img_path in tqdm.tqdm(glob.glob(input_location + 'testing_dataset/image/*.png'), desc="正在載入測試照片"):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        test_images.append(img)

    # 捕捉遮罩/標籤信息，以列表形式保存
    for mask_path in tqdm.tqdm(glob.glob(input_location + f'testing_dataset/{n_classes}_label/*.png'), desc="正在載入測試標籤"):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        test_masks.append(mask)

    # 將列表轉換為陣列，以供機器學習處理
    test_images = np.array(test_images)
    test_masks = np.array(test_masks)

    print("test image shape: " + str(test_images.shape))
    print("test mask shape: " + str(test_masks.shape))

    ###############################################
    # Encode labels... but multi dim array so need to flatten, encode and reshape
    label_encoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1, 1)
    train_masks_reshaped_encoded = label_encoder.fit_transform(train_masks_reshaped.ravel())
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    ###############################################
    train_images_norm = train_images / 255

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    ###del
    del train_masks_encoded_original_shape, train_masks_reshaped
    gc.collect()
    X_train, X_validation, y_train, y_validation = train_test_split(train_images_norm,
                                                        train_masks_input,
                                                        test_size=0.2,
                                                        random_state=random_state)
    # Print shapes of the sets
    print("Training set shape: ", X_train.shape)
    print("Validation set shape: ", X_validation.shape)
    del train_images_norm, train_masks_input
    gc.collect()
    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled
    #訓練類別
    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    #測試類別
    val_masks_cat = to_categorical(y_validation, num_classes=n_classes)
    y_val_cat = val_masks_cat.reshape((y_validation.shape[0], y_validation.shape[1], y_validation.shape[2], n_classes))

    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=np.unique(train_masks_reshaped_encoded),
                                         y=train_masks_reshaped_encoded).tolist()

    model = UNet(n_classes=n_classes, img_height=SIZE_X, img_width=SIZE_Y).model
    model.compile(optimizer='adam',
                  loss= weighted_loss(categorical_crossentropy, class_weights),
                  metrics=['accuracy'])

    model_name = f'model_rd_{random_state}.h5'
    save_path = os.path.join(output_location, model_name)

    learning_rate_function = ReduceLROnPlateau(monitor='val_loss',
                                               patience=3,
                                               verbose=1,
                                               factor=0.6,
                                               min_lr= learning_rate)
    early_stop = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=60,
                              verbose=1)
    model_check = ModelCheckpoint(save_path,
                                 monitor='val_loss',
                                 mode='min',
                                 verbose=1,
                                 save_best_only=True)
    train_history = model.fit(X_train, y_train_cat,
                        batch_size=batch_size,
                        verbose=1,
                        epochs=300,
                        validation_data=(X_validation, y_val_cat),
                        shuffle=False,
                        callbacks=[learning_rate_function, early_stop, model_check])
    draw_train_history_plot(train_history=train_history,
                            picture_name='loss&accuracy',
                            location=output_location)
    try:
        model.load_weights(save_path)
        print("載入模型成功!繼續訓練模型")
    except:
        print("載入模型失敗!開始訓練一個新模型")
    # evaluate model
    _, acc = model.evaluate(X_train, y_train_cat)
    print("Accuracy is = ", (acc * 100.0), "%")

    ############################################################################################
    #validation analysis
    # IOU
    y_validation_prediction = model.predict(X_validation)
    y_validation_prediction_argmax = np.argmax(y_validation_prediction, axis=3)

    # Using built-in keras function
    val_IOU_keras = MeanIoU(num_classes=n_classes)
    val_IOU_keras.update_state(y_validation[:, :, :, 0], y_validation_prediction_argmax)
    print("Val Mean IoU =", val_IOU_keras.result().numpy())

    val_values_cm = val_int_array = np.array(val_IOU_keras.get_weights()).astype(int)
    val_values_cm = val_values_cm.reshape(n_classes, n_classes)

    plot_confusion_matrix('validation_cm',
                          location=output_location,
                          cm=val_values_cm,
                          classes=target_names[n_classes])
    #validation_report
    validation_report = classification_report(y_validation[:, :, :, 0].ravel(),
                                               y_validation_prediction_argmax.ravel(),
                                               digits=4,
                                               target_names=target_names[n_classes],
                                               output_dict=True)
    ############################################################################################
    #test analysis
    n, h, w = test_masks.shape
    test_masks_reshaped = test_masks.reshape(-1, 1)
    test_masks_reshaped_encoded = label_encoder.fit_transform(test_masks_reshaped.ravel())
    test_masks_encoded_original_shape = test_masks_reshaped_encoded.reshape(n, h, w)

    test_images_norm = test_images / 255
    test_masks_input = np.expand_dims(test_masks_encoded_original_shape, axis=3)
    x_test = test_images_norm
    y_test = test_masks_input
    print("test Class values in the dataset are ... ", np.unique(test_masks_input))
    # IOU
    y_test_prediction = model.predict(x_test)
    y_test_prediction_argmax = np.argmax(y_test_prediction, axis=3)

    test_IOU_keras = MeanIoU(num_classes=n_classes)
    test_IOU_keras.update_state(y_test[:, :, :, 0], y_test_prediction_argmax)
    print("Test Mean IoU =", test_IOU_keras.result().numpy())

    test_values_cm = test_int_array = np.array(test_IOU_keras.get_weights()).astype(int)
    test_values_cm = test_values_cm.reshape(n_classes, n_classes)
    plot_confusion_matrix('test_cm',
                          location=output_location,
                          cm=test_values_cm,
                          classes=target_names[n_classes])
    #test_report
    print('test_masks',np.unique(y_test[:, :, :, 0].ravel()))
    print('y_test_prediction_argmax',np.unique(y_test_prediction_argmax.ravel()))

    testing_report = classification_report(y_test[:, :, :, 0].ravel(),
                                           y_test_prediction_argmax.ravel(),
                                           digits=4,
                                           target_names=target_names[n_classes],
                                           output_dict=True)
    ############################################################################################
    # 製作excel
    val_df = pd.DataFrame(validation_report).transpose()
    test_df = pd.DataFrame(testing_report).transpose()

    val_row_data, test_row_data = {}, {}
    for col in val_df.columns:
        val_row_data[col] = 'validation'
        test_row_data[col] = 'test'
    top_val_row = pd.DataFrame(val_row_data, index=[0])
    top_test_row = pd.DataFrame(test_row_data, index=[0])

    val_df = pd.concat([top_val_row, val_df], ignore_index=False, axis=0)
    test_df = pd.concat([top_test_row, test_df], ignore_index=False, axis=0)
    result_df = pd.concat([val_df, test_df], ignore_index=False, axis=0)

    # 检查文件是否存在
    excel_file_path = f'{excel_output_location}result.xlsx'
    if os.path.isfile(excel_file_path):
        # 如果文件已存在，在現有 Excel 檔案的指定工作表中寫入 DataFrame
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
            result_df.to_excel(writer, index=True, sheet_name=f'random_{random_state}')
            print('已寫入文件')
    else:
        # 如果文件不存在，創建新的 Excel 文件並寫入 DataFrame
        result_df.to_excel(excel_file_path, index=True, sheet_name=f'random_{random_state}')
        print('已寫入文件')
    print('訓練完成!!')