import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from sklearn.preprocessing import LabelEncoder
from db_connector import *
from tensorflow.keras.preprocessing.sequence import pad_sequences


def scheduler(epoch, lr):
    if epoch < 15:
        lr = 1e-3
        return lr
    else:
        lr = 1e-4
        return lr


def scheduler2(epoch, lr):
    if epoch < 3:
        lr = 1e-3
        return lr
    else:
        lr = 1e-4
        return lr


def neural_network_brand_mapping():
    option = input('1 - Train Model 1 DNN (BU Code + BU Brand ---> IB Brand (train from IB Brand Master + IBBI)\n'
                   '2 - Train Model 2 DNN (BU Code + BU Brand ---> IB Brand (train from IBBI)\n'
                   '3 - Train Model 3 DNN (BU Code + BU Brand + BU Supplier ---> Mother Company (train from IBBI)\n'
                   '4 - Monthly IBBI Data Check\n'
                   '5 - Bundle BU Brand Name Check\n'
                   'Function you want: ')
    option = int(option)

    """----------------------------IBBI--------------------------"""
    IBBI_dataset = pd.read_excel('bu_monthly_checking_previous_month.xlsx', sheet_name='Report 1', dtype=str)
    IBBI_dataset.dropna(subset=['BU Brand Name'], inplace=True)
    IBBI_dataset.drop_duplicates(subset=['BU Code', 'BU Brand Name', 'BU Supplier Name'], inplace=True, keep='last')
    IBBI_dataset = IBBI_dataset.astype(str)
    IBBI_dataset.loc[IBBI_dataset['Mother Company Name'] != 'OTHER SUPPLIER', 'Remarks'] = 'Mapped Before'
    IBBI_dataset.loc[IBBI_dataset['Mother Company Name'] == 'OTHER SUPPLIER', 'Remarks'] = 'OTHER SUPPLIER'
    IBBI_dataset.loc[IBBI_dataset['BU Brand Name'] != IBBI_dataset['IB Brand Name'], 'Remarks'] = 'Mapped Before'

    bo_dataset = IBBI_dataset[IBBI_dataset['Remarks'] == 'Mapped Before']
    x_bo_dataset = bo_dataset.loc[:, ['BU Code', 'BU Brand Name', 'BU Supplier Name']]
    y_bo_dataset = bo_dataset.loc[:, 'IB Brand Name']                           ###
    bo_supplier_tokenizer = Tokenizer()
    bo_supplier_tokenizer.fit_on_texts(x_bo_dataset['BU Supplier Name'])
    bo_IB_brand_name_label_encoder = LabelEncoder()
    bo_IB_brand_name_label_encoder.fit(y_bo_dataset)

    # Train Supplier / Mother Company Prediction
    bo2_dataset = IBBI_dataset[IBBI_dataset['Mother Company Name'] != 'OTHER SUPPLIER']
    x_bo2_dataset = bo2_dataset.loc[:, ['BU Code', 'BU Brand Name', 'BU Supplier Name']]
    y_bo2_dataset = bo2_dataset.loc[:, 'Mother Company Name']                   ###
    x_train2_IBBI, x_test2_IBBI, y_train_IBBI_supplier, y_test_IBBI_supplier = train_test_split(x_bo2_dataset, y_bo2_dataset, test_size=0.1, random_state=0)
    bo_supplier_label_encoder = LabelEncoder()
    bo_supplier_label_encoder.fit(y_bo2_dataset)

    """----------------------IB Brand Master-----------------------"""
    occ = 0
    # IB_Brand_Master = pd.read_excel(r'T:/Group File/Department/IB_RH/BI Project/BI Mapping/Brand List/IB BRAND/IB Brand Mapping 2018.xlsx', sheet_name='Brand Master')
    IB_Brand_Master = pd.read_excel('IB Brand Mapping 2018.xlsx', sheet_name='Brand Master')
    IB_Brand_Master.dropna(subset=['BU Brand Name'], inplace=True)
    IB_Brand_Master.drop_duplicates(subset=['BU Code', 'BU Brand Name'], keep='last', inplace=True)
    IB_Brand_Master = IB_Brand_Master.tail(30000)
    IB_Brand_Master = IB_Brand_Master.astype(str)
    # IB_Brand_Master = IB_Brand_Master[IB_Brand_Master.groupby('IB Brand Name')['IB Brand Name'].transform('size') > occ]

    ### IB Brand Master + bo_dataset
    IB_Brand_Master = pd.concat([IB_Brand_Master, bo_dataset])
    IB_Brand_Master.drop_duplicates(subset=['BU Code', 'BU Brand Name'], keep='last', inplace=True)
    IB_Brand_Master = IB_Brand_Master[IB_Brand_Master['IB Brand Name'] != '-']

    # x = IB_Brand_Master.loc[:, ['BU Code', 'BU Brand Name']]
    # y = IB_Brand_Master['IB Brand Name']
    x = bo_dataset.loc[:, ['BU Code', 'BU Brand Name']]
    y = bo_dataset['IB Brand Name']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=0)
    brand_tokenizer = Tokenizer( char_level=True)
    brand_tokenizer.fit_on_texts(x['BU Brand Name'])
    bu_tokenizer = Tokenizer(num_words=43)
    bu_tokenizer.fit_on_texts(x['BU Code'])
    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    maxlen_brand = 58
    maxlen_supplier = 10

    # BU Brand ---> IB Brand (train from IB Brand Master)
    if option == 1:

        x_train_bu = bu_tokenizer.texts_to_matrix(x_train['BU Code'], mode='binary')
        x_test_bu = bu_tokenizer.texts_to_matrix(x_test['BU Code'], mode='binary')
        x_train_tfidf = brand_tokenizer.texts_to_matrix(x_train['BU Brand Name'], mode='tfidf')
        x_test_tfidf = brand_tokenizer.texts_to_matrix(x_test['BU Brand Name'], mode='tfidf')
        y_train_encode = label_encoder.transform(y_train)
        y_test_encode = label_encoder.transform(y_test)
        no_of_label = len(y.value_counts())
        print(f'Training set: {x_train_tfidf.shape} | Test set: {x_test_tfidf.shape}\n')
        print(f'IB Brand Master shape: {IB_Brand_Master.shape} | no. of vocab (brand): {len(brand_tokenizer.word_index)} | no. of label: {no_of_label}\n')

        inputA = Input(batch_shape=(None, x_train_bu.shape[1]))
        inputB = Input(batch_shape=(None, x_train_tfidf.shape[1]))

        one = Dense(32, activation='relu')(inputA)
        one = Dropout(0.5)(one)
        one = Model(inputs=inputA, outputs=one)

        two = Dense(500, activation='relu')(inputB)
        two = Dropout(0.2)(two)
        two = Dense(400, activation='relu')(two)
        two = Model(inputs=inputB, outputs=two)

        combined = concatenate([one.output, two.output])
        combined = BatchNormalization()(combined)
        combined = Dense(400, activation='relu')(combined)
        combined = Dense(no_of_label, activation='softmax')(combined)

        model = Model(inputs=[one.input, two.input], outputs=combined)
        model.summary()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.fit([x_train_bu, x_train_tfidf], y_train_encode, epochs=8, batch_size=128,
                  validation_data=([x_test_bu, x_test_tfidf], y_test_encode))

        model.save(f'saved_model/NN_Brand_Mapping_model1_DNN.h5')
        model.evaluate([x_test_bu, x_test_tfidf], y_test_encode)


    # Brand Prediction (CNN)

    elif option == 2:

        x_train_bu = bu_tokenizer.texts_to_matrix(x_train['BU Code'], mode='binary')
        x_test_bu = bu_tokenizer.texts_to_matrix(x_test['BU Code'], mode='binary')
        x_train = brand_tokenizer.texts_to_sequences(x_train['BU Brand Name'])
        x_test = brand_tokenizer.texts_to_sequences(x_test['BU Brand Name'])
        x_train = pad_sequences(x_train, padding='post')
        maxlen = x_train.shape[1]           #58
        x_test = pad_sequences(x_test, padding='post',maxlen=maxlen)
        y_train_encode = label_encoder.transform(y_train)
        y_test_encode = label_encoder.transform(y_test)
        no_of_label = len(y.value_counts())

        V = len(brand_tokenizer.word_index)     #181
        D = 64                                 #500
        T = 40

        print(f'V: {V} | D: {D} | T: {T}')
        print(f'Training set: {x_train.shape} | Test set: {x_test.shape}\n')
        print(f'IB Brand Master shape: {IB_Brand_Master.shape} | no. of vocab (brand): {len(brand_tokenizer.word_index)} | no. of label: {no_of_label}\n')

        inputA = Input(batch_shape=(None, x_train_bu.shape[1]))
        inputB = Input(batch_shape=(None, x_train.shape[1]))

        one = Dense(32, activation='relu')(inputA)
        one = Dropout(0.5)(one)
        one = Model(inputs=inputA, outputs=one)

        two = Embedding(input_dim=V+1, output_dim=D, input_length=T, trainable=True)(inputB)
        two = Conv1D(filters=128, kernel_size=10, activation='relu', padding='same')(two)
        two = MaxPooling1D(4, padding='same')(two)
        two = Dropout(0.3)(two)
        two = Conv1D(filters=128, kernel_size=8, activation='relu', padding='same')(two)
        two = MaxPooling1D(4, padding='same')(two)
        two = Dropout(0.3)(two)
        two = GlobalMaxPooling1D()(two)
        two = Model(inputs=inputB, outputs=two)

        combined = concatenate([one.output, two.output])
        combined = BatchNormalization()(combined)
        combined = Dense(no_of_label, activation='softmax')(combined)

        model = Model(inputs=[one.input, two.input], outputs=combined)
        model.summary()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.fit([x_train_bu, x_train], y_train_encode, epochs=12, batch_size=64,
                  validation_data=([x_test_bu, x_test], y_test_encode))

        model.save(f'saved_model/NN_Brand_Mapping_model2_CNN.h5')
        model.evaluate([x_test_bu, x_test], y_test_encode)


    # BU Brand + BU Supplier ---> Mother Company (train from IBBI)
    elif option == 3:

        x_train_bu = bu_tokenizer.texts_to_matrix(x_train2_IBBI['BU Code'], mode='binary')
        x_test_bu = bu_tokenizer.texts_to_matrix(x_test2_IBBI['BU Code'], mode='binary')
        x_train_supplier = bo_supplier_tokenizer.texts_to_sequences(x_train2_IBBI['BU Supplier Name'])
        x_test_supplier = bo_supplier_tokenizer.texts_to_sequences(x_test2_IBBI['BU Supplier Name'])

        x_train_supplier = pad_sequences(x_train_supplier, padding='post')
        maxlen = x_train_supplier.shape[1]          #9
        x_test_supplier = pad_sequences(x_test_supplier,  padding='post', maxlen=maxlen)

        y_train_encode = bo_supplier_label_encoder.transform(y_train_IBBI_supplier)
        y_test_encode = bo_supplier_label_encoder.transform(y_test_IBBI_supplier)
        no_of_label = len(y_bo2_dataset.value_counts())

        V = len(bo_supplier_tokenizer.word_index)
        D = 64
        T = 5

        print(bo2_dataset['Mother Company Name'].value_counts()) #IBBI known: {mo_known.shape} | unknown: {mo_unknown.shape}\n
        print(f'Training set: {x_train2_IBBI.shape} | Test set: {x_test2_IBBI.shape}\n')
        print(f'IBBI shape: {bo2_dataset.shape} | no. of vocab (supplier): {len(bo_supplier_tokenizer.word_index)} | no. of label: {no_of_label} \n')
        print(f'V: {V} | D: {D} | T: {T}')

        input_BU = Input(batch_shape=(None, x_train_bu.shape[1]))
        inputB = Input(batch_shape=(None, x_train_supplier.shape[1]))

        bu = Dense(32, activation='relu')(input_BU)
        bu = Dropout(0.5)(bu)
        bu = Model(inputs=input_BU, outputs=bu)

        two = Embedding(input_dim=V+1, output_dim=D, input_length=T)(inputB)
        two = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(two)
        two = MaxPooling1D(1, padding='same')(two)
        two = Dropout(0.3)(two)
        two = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(two)
        two = MaxPooling1D(1, padding='same')(two)
        two = Dropout(0.3)(two)
        two = GlobalMaxPooling1D()(two)
        two = Model(inputs=inputB, outputs=two)

        combined = concatenate([bu.output, two.output])
        combined = BatchNormalization()(combined)
        combined = Dense(no_of_label, activation="softmax")(combined)

        model = Model(inputs=[bu.input, two.input], outputs=combined)
        model.summary()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.fit(x=[x_train_bu,  x_train_supplier], y=y_train_encode, epochs=8, batch_size=32,
                  validation_data=([x_test_bu, x_test_supplier], y_test_encode))
        model.save('saved_model/NN_Supplier_Mapping_model3_CNN.h5')
        model.evaluate(x=[x_test_bu, x_test_supplier], y=y_test_encode)


    elif option == 4:
        print("Program running........")
        model1 = tf.keras.models.load_model('saved_model/NN_Brand_Mapping_model2_CNN.h5')
        model3 = tf.keras.models.load_model('saved_model/NN_Supplier_Mapping_model3_CNN.h5')

        # monthly_data = run_query(bu_monthly_checking)
        monthly_data = pd.read_excel('bu_monthly_checking_current_month.xlsx', sheet_name='Report 1', dtype=str)
        monthly_data = monthly_data.astype(str)
        cursor = 0
        y_temp_brand = []
        y_temp_supplier = []
        y_temp_supplier_prob =[]
        y_temp_brand_prob = []

        while cursor < monthly_data.last_valid_index():
            x_bu = monthly_data.loc[cursor:cursor + 10000, 'BU Code']
            x_brand = monthly_data.loc[cursor:cursor + 10000, 'BU Brand Name']
            x_sup = monthly_data.loc[cursor:cursor + 10000, 'BU Supplier Name']

            x_bu_ibbm = bu_tokenizer.texts_to_matrix(x_bu, mode='binary')
            # x_brand_tfidf = brand_tokenizer.texts_to_matrix(x_brand, mode='tfidf')
            x_brand = brand_tokenizer.texts_to_sequences(x_brand)
            x_brand = pad_sequences(x_brand, padding='post', maxlen=maxlen_brand)
            x_supplier = bo_supplier_tokenizer.texts_to_sequences(x_sup)
            x_supplier = pad_sequences(x_supplier, padding='post', maxlen=maxlen_supplier)

            # Brand prediction 1 (2 Feature)
            y_pred_brand_name = model1.predict([x_bu_ibbm, x_brand])
            y_pred_brand = np.argmax(y_pred_brand_name, axis=1)
            y_pred_brand_prob = np.max(y_pred_brand_name, axis=1)
            y_temp_brand_prob.extend(y_pred_brand_prob)
            y_pred_brand = label_encoder.inverse_transform(y_pred_brand.tolist())
            y_temp_brand.extend(y_pred_brand)

            # Supplier prediction (3 Features)
            y_pred_supplier_name = model3.predict([x_bu_ibbm, x_supplier])
            y_pred_supplier = np.argmax(y_pred_supplier_name, axis=1)
            y_pred_supplier_prob = np.max(y_pred_supplier_name, axis=1)
            y_temp_supplier_prob.extend(y_pred_supplier_prob)
            y_pred_supplier = bo_supplier_label_encoder.inverse_transform(y_pred_supplier.tolist())
            y_temp_supplier.extend(y_pred_supplier)
            cursor += 10001

        y_temp_brand = pd.DataFrame(y_temp_brand, columns=['IB Brand Name (Prediction 1)'])
        y_temp_brand['IB Brand Name (Prediction 1)'] = y_temp_brand['IB Brand Name (Prediction 1)'].str.upper()
        y_temp_supplier = pd.DataFrame(y_temp_supplier, columns=['Mother Company (Prediction)'])

        y_temp_brand_prob = pd.DataFrame(y_temp_brand_prob, columns=['Brand (Prob)'])
        y_temp_supplier_prob = pd.DataFrame(y_temp_supplier_prob, columns=['Supplier (Prob)'])


        result = pd.concat([monthly_data, y_temp_brand, y_temp_supplier, y_temp_brand_prob, y_temp_supplier_prob], axis=1)  # y_temp_brand2
        result.loc[result['Mother Company Name'] != 'OTHER SUPPLIER', 'Remarks'] = 'Mapped Before'
        result.loc[result['Mother Company Name'] == 'OTHER SUPPLIER', 'Remarks'] = 'Other Supplier'
        result.loc[result['BU Brand Name'] != result['IB Brand Name'], 'Remarks'] = 'Mapped Before'
        result.loc[(result['IB Brand Name'] == result['IB Brand Name (Prediction 1)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Accuracy1'] = 'Correct'
        result.loc[(result['IB Brand Name'] != result['IB Brand Name (Prediction 1)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Accuracy1'] = 'Incorrect'
        result.loc[(result['Mother Company Name'] == result['Mother Company (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Mo_Accuracy'] = 'Correct'
        result.loc[(result['Mother Company Name'] != result['Mother Company (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Mo_Accuracy'] = 'Incorrect'

        # with pd.ExcelWriter('Output/bu_monthly_checking_NN_RESULT.xlsx', mode='a') as writer:
        #     existing = pd.read_excel('Output/bu_monthly_checking_NN_RESULT.xlsx')
        #     existing = existing['BU Code']
        #     existing.drop_duplicates(inplace=True)
        #     result = result[result['BU Code']!=existing['BU Code']]
        #     result.to_excel(writer, index=False, header=None)
        result.to_excel('Output/bu_monthly_checking_NN_RESULT.xlsx', index=False)

        acc1 = result.query('Accuracy1 == "Correct"').shape[0] / result[~result['Accuracy1'].isna()].shape[0] * 100
        acc3 = result.query('Mo_Accuracy == "Correct"').shape[0] / result[~result['Mo_Accuracy'].isna()].shape[0] * 100
        print(f'Accuracy1: {acc1}\nMother Company Accuracy: {acc3}')

    elif option == 5:
        model = tf.keras.models.load_model('saved_model/NN_Brand_Mapping_1.h5')
        bu_bundle = ['KV', 'ICI', 'HK', 'UK']
        with pd.ExcelWriter('brand_bundle_check/bundle_check_result.xlsx') as writer:
            for bu in bu_bundle:
                dataset = pd.read_excel('brand_bundle_check/data.xlsx', sheet_name=bu)
                dataset = dataset.astype(str)
                unique = dataset.groupby('BU Brand Name')['IB Brand Name'].unique().transform(len).rename('unique')
                dataset = dataset.merge(unique, on='BU Brand Name', how='left')
                x_real = dataset.loc[:, 'BU Brand Name']
                x_real_tfidf = brand_tokenizer.texts_to_matrix(x_real, mode='tfidf')
                y_pred = model.predict(x_real_tfidf)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = label_encoder.inverse_transform(y_pred.tolist())
                y_pred = pd.DataFrame(y_pred, columns=['IB Brand Name (Prediction 1)'])
                result = pd.concat([dataset, y_pred], axis=1)
                result = result[result['unique'] > 1]
                result = result.sort_values(by=['BU Brand Name'])
                result.to_excel(writer, sheet_name=bu, index=False)

    elif option == 6:      ##old code

        print("Program running........")
        model1 = tf.keras.models.load_model('saved_model/NN_Brand_Mapping_model1_DNN.h5')
        model2 = tf.keras.models.load_model('saved_model/NN_Brand_Mapping_model2_DNN.h5')
        model3 = tf.keras.models.load_model('saved_model/NN_Supplier_Mapping_model3_DNN.h5')
        monthly_data = pd.read_excel('bu_monthly_checking_current_month.xlsx', sheet_name='Report 1', dtype=str)
        monthly_data = monthly_data.astype(str)
        monthly_data['BU Brand Name'].fillna('', inplace=True)
        monthly_data.reset_index(drop=True, inplace=True)

        cursor = 0
        y_temp_brand = []
        y_temp_brand2 = []
        y_temp_supplier = []

        while cursor < monthly_data.last_valid_index():
            x_bu = monthly_data.loc[cursor:cursor + 10000, 'BU Code']
            x_brand = monthly_data.loc[cursor:cursor + 10000, 'BU Brand Name']
            x_sup = monthly_data.loc[cursor:cursor + 10000, 'BU Supplier Name']

            x_bu_ibbm = bu_tokenizer.texts_to_matrix(x_bu, mode='binary')
            x_brand_tfidf = brand_tokenizer.texts_to_matrix(x_brand, mode='tfidf')
            x_brand_for_supplier_tfidf = bo_brand_tokenizer.texts_to_matrix(x_brand, mode='tfidf')
            x_supplier_tfidf = bo_supplier_tokenizer.texts_to_matrix(x_sup, mode='tfidf')

            # Brand prediction 1 (2 Feature)
            y_pred_brand = model1.predict([x_bu_ibbm, x_brand_tfidf])
            y_pred_brand = np.argmax(y_pred_brand, axis=1)
            y_pred_brand = label_encoder.inverse_transform(y_pred_brand.tolist())
            y_temp_brand.extend(y_pred_brand)

            # # Brand prediction 2 (2 Features)
            # y_pred_brand2 = model2.predict([x_bu_ibbm, x_brand_for_supplier_tfidf])
            # y_pred_brand2 = np.argmax(y_pred_brand2, axis=1)
            # y_pred_brand2 = bo_IB_brand_name_label_encoder.inverse_transform(y_pred_brand2.tolist())
            # y_temp_brand2.extend(y_pred_brand2)

            # Supplier prediction (3 Features)
            y_pred_supplier = model3.predict([x_bu_ibbm,  x_brand_for_supplier_tfidf, x_supplier_tfidf])
            y_pred_supplier = np.argmax(y_pred_supplier, axis=1)
            y_pred_supplier = bo_supplier_label_encoder.inverse_transform(y_pred_supplier.tolist())
            y_temp_supplier.extend(y_pred_supplier)
            cursor += 10001

        y_temp_brand = pd.DataFrame(y_temp_brand, columns=['IB Brand Name (Prediction 1)'])
        y_temp_brand['IB Brand Name (Prediction 1)'] = y_temp_brand['IB Brand Name (Prediction 1)'].str.upper()
        # y_temp_brand2 = pd.DataFrame(y_temp_brand2, columns=['IB Brand Name (Prediction 2)'])
        # y_temp_brand2['IB Brand Name (Prediction 2)'] = y_temp_brand2['IB Brand Name (Prediction 2)'].str.upper()
        y_temp_supplier = pd.DataFrame(y_temp_supplier, columns=['Mother Company (Prediction)'])

        result = pd.concat([monthly_data, y_temp_brand,  y_temp_supplier], axis=1)                     # y_temp_brand2
        result.loc[result['Mother Company Name'] != 'OTHER SUPPLIER', 'Remarks'] = 'Mapped Before'
        result.loc[result['Mother Company Name'] == 'OTHER SUPPLIER', 'Remarks'] = 'Other Supplier'
        result.loc[result['BU Brand Name'] != result['IB Brand Name'], 'Remarks'] = 'Mapped Before'
        result.loc[(result['IB Brand Name'] == result['IB Brand Name (Prediction 1)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Accuracy1'] = 'Correct'
        result.loc[(result['IB Brand Name'] != result['IB Brand Name (Prediction 1)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Accuracy1'] = 'Incorrect'
        # result.loc[(result['IB Brand Name'] == result['IB Brand Name (Prediction 2)']) &
        #            (result['Remarks'] == 'Mapped Before'), 'Accuracy2'] = 'Correct'
        # result.loc[(result['IB Brand Name'] != result['IB Brand Name (Prediction 2)']) &
        #            (result['Remarks'] == 'Mapped Before'), 'Accuracy2'] = 'Incorrect'
        result.loc[(result['Mother Company Name'] == result['Mother Company (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Mo_Accuracy'] = 'Correct'
        result.loc[(result['Mother Company Name'] != result['Mother Company (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Mo_Accuracy'] = 'Incorrect'

        # result = pd.concat([result, y_temp_supplier], axis=1)

        _005C = pd.read_excel('005ABCD/005C_Mother_Company.xlsx', sheet_name='Sheet1')
        _005C = _005C.iloc[:, [1, 4, 6]]
        _005C.rename(columns={'Mother Company': 'Mother Company (Prediction)'}, inplace=True)
        result = result.merge(_005C, on='Mother Company (Prediction)', how='left')

        result.to_excel('Output/bu_monthly_checking_NN_RESULT.xlsx', index=False)
        acc1 = result.query('Accuracy1 == "Correct"').shape[0] / result[~result['Accuracy1'].isna()].shape[0] * 100
        acc2 = result.query('Accuracy2 == "Correct"').shape[0] / result[~result['Accuracy2'].isna()].shape[0] * 100
        acc3 = result.query('Mo_Accuracy == "Correct"').shape[0] / result[~result['Mo_Accuracy'].isna()].shape[0] * 100
        print(f'Accuracy1: {acc1}\nAccuracy2: {acc2}\nMother Company Accuracy: {acc3}')


neural_network_brand_mapping()

