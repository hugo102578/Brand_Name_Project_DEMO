import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, Embedding, Conv1D, AveragePooling1D, BatchNormalization, Flatten, MaxPooling1D, GlobalMaxPooling1D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Training_Source import Training_Source
from db_connector import SQL




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


class Neural_Network_Brand_Mapping:

    def __init__(self, db=None):
        self.db = db
        training_source = Training_Source(self.db)
        bo_dataset1 = training_source.get_brand_training_source()
        self.x_bo_dataset1 = bo_dataset1.loc[:, ['BU Code', 'BU Brand Name']]
        self.y_bo_dataset1 = bo_dataset1['IB Brand Name']
        self.x_train1, self.x_test1, self.y_train1, self.y_test1 = train_test_split(self.x_bo_dataset1, self.y_bo_dataset1, test_size=0.05, shuffle=True, random_state=0)
        self.brand_tokenizer = Tokenizer(char_level=True)
        self.brand_tokenizer.fit_on_texts(self.x_bo_dataset1['BU Brand Name'])
        self.bu_tokenizer = Tokenizer()
        self.bu_tokenizer.fit_on_texts(self.x_bo_dataset1['BU Code'])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y_bo_dataset1)

        # Mother Company pre-processing
        bo_dataset2 = training_source.get_supplier_training_source()
        self.x_bo_dataset2 = bo_dataset2.loc[:, ['BU Code', 'BU Supplier Name']]
        self.y_bo_dataset2 = bo_dataset2.loc[:, 'Mother Company Name']
        self.x_train2, self.x_test2, self.y_train2, self.y_test2 = train_test_split(self.x_bo_dataset2, self.y_bo_dataset2, test_size=0.05, random_state=0)
        self.bo_supplier_tokenizer = Tokenizer()
        self.bo_supplier_tokenizer.fit_on_texts(self.x_bo_dataset2['BU Supplier Name'])
        self.bo_supplier_label_encoder = LabelEncoder()
        self.bo_supplier_label_encoder.fit(self.y_bo_dataset2)

        self.maxlen_brand = 58
        self.maxlen_supplier = 10
        self.SQL = SQL()

    def train_CNN_brand(self):

        print('Start Training Brand Mapping Model......')
        x_train_bu = self.bu_tokenizer.texts_to_matrix(self.x_train1['BU Code'], mode='binary')
        x_test_bu = self.bu_tokenizer.texts_to_matrix(self.x_test1['BU Code'], mode='binary')
        x_train_brand = self.brand_tokenizer.texts_to_sequences(self.x_train1['BU Brand Name'])
        x_test_brand = self.brand_tokenizer.texts_to_sequences(self.x_test1['BU Brand Name'])
        x_train_brand = pad_sequences(x_train_brand, padding='post')
        maxlen = x_train_brand.shape[1]  # 58
        x_test_brand = pad_sequences(x_test_brand, padding='post', maxlen=maxlen)
        y_train_encode = self.label_encoder.transform(self.y_train1)
        y_test_encode = self.label_encoder.transform(self.y_test1)
        no_of_label = len(self.y_bo_dataset1.value_counts())

        V = len(self.brand_tokenizer.word_index)  # 122
        D = 64
        T = 40

        print(f'V: {V} | D: {D} | T: {T}\n')
        print(f'Training set: {self.x_train1.shape} | Test set: {self.x_test1.shape}\nNo. of label: {no_of_label}\n')

        inputA = Input(batch_shape=(None, x_train_bu.shape[1]))
        inputB = Input(batch_shape=(None, x_train_brand.shape[1]))

        one = Dense(64, activation='relu')(inputA)
        one = Dropout(0.5)(one)
        one = Dense(32, activation='relu')(one)
        one = Model(inputs=inputA, outputs=one)

        two = Embedding(input_dim=V + 1, output_dim=D, input_length=T)(inputB)
        two = Conv1D(filters=128, kernel_size=10, activation='relu', padding='causal')(two)
        two = Conv1D(filters=128, kernel_size=8, activation='relu', padding='causal')(two)
        two = AveragePooling1D(4, padding='same')(two)
        two = BatchNormalization()(two)
        two = Flatten()(two)
        two = Dropout(0.5)(two)
        two = Dense(512, activation='relu')(two)
        two = Model(inputs=inputB, outputs=two)

        combined = concatenate([one.output, two.output])
        combined = BatchNormalization()(combined)
        combined = Dense(no_of_label, activation='softmax')(combined)

        model = Model(inputs=[one.input, two.input], outputs=combined)
        model.summary()
        model.compile(loss=SparseCategoricalCrossentropy(),
                      optimizer=Adam(),
                      metrics=['accuracy'])
        model.fit([x_train_bu, x_train_brand], y_train_encode, epochs=10, batch_size=128,
                  validation_data=([x_test_bu, x_test_brand], y_test_encode))

        model.save(f'saved_model/NN_Brand_Mapping_model_CNN.h5')
        model.evaluate([x_test_bu, x_test_brand], y_test_encode)
        print('Training Finished.\n')

    def train_CNN_supplier(self):
        print('Start Training Supplier Mapping Model......')
        x_train_bu = self.bu_tokenizer.texts_to_matrix(self.x_train2['BU Code'], mode='binary')
        x_test_bu = self.bu_tokenizer.texts_to_matrix(self.x_test2['BU Code'], mode='binary')
        x_train_supplier = self.bo_supplier_tokenizer.texts_to_sequences(self.x_train2['BU Supplier Name'])
        x_test_supplier = self.bo_supplier_tokenizer.texts_to_sequences(self.x_test2['BU Supplier Name'])

        x_train_supplier = pad_sequences(x_train_supplier, padding='post')
        maxlen = x_train_supplier.shape[1]  # 9
        x_test_supplier = pad_sequences(x_test_supplier, padding='post', maxlen=maxlen)

        y_train_encode = self.bo_supplier_label_encoder.transform(self.y_train2)
        y_test_encode = self.bo_supplier_label_encoder.transform(self.y_test2)
        no_of_label = len(self.y_bo_dataset2.value_counts())

        V = len(self.bo_supplier_tokenizer.word_index)
        D = 64
        T = 5

        print(f'V: {V} | D: {D} | T: {T}')
        print(f'Training set: {self.x_train2.shape} | Test set: {self.x_test2.shape} | No. of label: {no_of_label}\n')

        input_BU = Input(batch_shape=(None, x_train_bu.shape[1]))
        inputB = Input(batch_shape=(None, x_train_supplier.shape[1]))

        bu = Dense(32, activation='relu')(input_BU)
        bu = Dropout(0.5)(bu)
        bu = Model(inputs=input_BU, outputs=bu)

        two = Embedding(input_dim=V + 1, output_dim=D, input_length=T)(inputB)
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
        model.compile(loss=SparseCategoricalCrossentropy(),
                      optimizer=Adam(),
                      metrics=['accuracy'])
        model.fit(x=[x_train_bu, x_train_supplier], y=y_train_encode, epochs=20, batch_size=32,
                  validation_data=([x_test_bu, x_test_supplier], y_test_encode))
        model.save('saved_model/NN_Supplier_Mapping_model_CNN.h5')
        model.evaluate(x=[x_test_bu, x_test_supplier], y=y_test_encode)
        print('Training Finished.\n')

    def predict_brand(self):

        model1 = load_model('saved_model/NN_Brand_Mapping_model_CNN.h5')
        model3 = load_model('saved_model/NN_Supplier_Mapping_model_CNN.h5')
        monthly_data = self.db.get_data(self.SQL.bu_monthly_checking_current_month).astype(str)
        monthly_data.loc[monthly_data['Net Sales Value - Lccy'].isin(['nan', '0.0']), 'Sales Flag'] = 'No Sales'
        monthly_data.loc[~monthly_data['Net Sales Value - Lccy'].isin(['nan', '0.0']), 'Sales Flag'] = 'Have Sales'
        print(f'Predicting Current Month ({self.SQL.current_month}) Brand And Supplier......')

        cursor = 0
        step = 50000
        y_temp_brand = []
        y_temp_supplier = []
        y_temp_supplier_prob = []
        y_temp_brand_prob = []


        while cursor < monthly_data.last_valid_index():
            x_bu = monthly_data.loc[cursor:cursor + step, 'BU Code']
            x_brand = monthly_data.loc[cursor:cursor + step, 'BU Brand Name']
            x_sup = monthly_data.loc[cursor:cursor + step, 'BU Supplier Name']

            x_bu = self.bu_tokenizer.texts_to_matrix(x_bu)
            x_brand = self.brand_tokenizer.texts_to_sequences(x_brand)
            x_brand = pad_sequences(x_brand, padding='post', maxlen=self.maxlen_brand)
            x_supplier = self.bo_supplier_tokenizer.texts_to_sequences(x_sup)
            x_supplier = pad_sequences(x_supplier, padding='post', maxlen=self.maxlen_supplier)

            y_pred_brand_name = model1.predict([x_bu, x_brand])
            y_pred_brand_prob = np.max(y_pred_brand_name, axis=1)
            y_temp_brand_prob.extend(y_pred_brand_prob)
            y_pred_brand = np.argmax(y_pred_brand_name, axis=1)
            y_pred_brand = self.label_encoder.inverse_transform(y_pred_brand.tolist())
            y_temp_brand.extend(y_pred_brand)

            y_pred_supplier_name = model3.predict([x_bu, x_supplier])
            y_pred_supplier = np.argmax(y_pred_supplier_name, axis=1)
            y_pred_supplier_prob = np.max(y_pred_supplier_name, axis=1)
            y_temp_supplier_prob.extend(y_pred_supplier_prob)
            y_pred_supplier = self.bo_supplier_label_encoder.inverse_transform(y_pred_supplier.tolist())
            y_temp_supplier.extend(y_pred_supplier)
            cursor += step+1

        y_temp = pd.DataFrame({'IB Brand Name (Prediction)': y_temp_brand, 'Mother Company (Prediction)': y_temp_supplier,
                               'Brand (Prob)': y_temp_brand_prob, 'Supplier (Prob)': y_temp_supplier_prob})

        result = pd.concat([monthly_data, y_temp], axis=1)
        result.loc[result['Mother Company Name'] != 'OTHER SUPPLIER', 'Remarks'] = 'Mapped Before'
        result.loc[result['Mother Company Name'] == 'OTHER SUPPLIER', 'Remarks'] = 'Other Supplier'
        result.loc[result['BU Brand Name'] != result['IB Brand Name'], 'Remarks'] = 'Mapped Before'
        result.loc[(result['IB Brand Name'] == result['IB Brand Name (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Accuracy1'] = 'Correct'
        result.loc[(result['IB Brand Name'] != result['IB Brand Name (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Accuracy1'] = 'Incorrect'
        result.loc[(result['Mother Company Name'] == result['Mother Company (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Mo_Accuracy'] = 'Correct'
        result.loc[(result['Mother Company Name'] != result['Mother Company (Prediction)']) &
                   (result['Remarks'] == 'Mapped Before'), 'Mo_Accuracy'] = 'Incorrect'
        result.loc[result['Brand (Prob)'] < 0.6, 'IB Brand Name (Prediction)'] = 'UNKNOWN'
        result.loc[result['Supplier (Prob)'] < 0.6, 'Mother Company (Prediction)'] = 'UNKNOWN'

        with pd.ExcelWriter('Output/bu_monthly_checking_NN_Prediction.xlsx', mode='w', engine='openpyxl') as writer:
            result.to_excel(writer, index=False, sheet_name='Sheet1')
            checked_BU = result['BU Code'].drop_duplicates()
            checked_BU.to_excel(writer, index=False, sheet_name='checked_BU')

        acc1 = result.query('Accuracy1 == "Correct"').shape[0] / result[~result['Accuracy1'].isna()].shape[0] * 100
        acc3 = result.query('Mo_Accuracy == "Correct"').shape[0] / result[~result['Mo_Accuracy'].isna()].shape[0] * 100
        print(f'Prediction Finished.\nBrand Accuracy: {acc1:.2f}%\nMother Company Accuracy: {acc3:.2f}%\n')

    def brand_master_bot(self):
        import warnings
        warnings.simplefilter("ignore")
        print('Connecting Prediction To Brand Master (A-Brand, Luxury, Exclusive)......')
        pt_input = None
        pred = 'IB Brand Name (Prediction)'
        bu_display_name = pd.read_excel('BU_display_name.xlsx')
        bu_display_name = bu_display_name.rename(columns={'BU Code': 'BU_Code'})
        nn_result = pd.read_excel('Output/bu_monthly_checking_NN_Prediction.xlsx')
        nn_result = nn_result[nn_result[pred] != 'UNKNOWN']
        nn_result = nn_result[nn_result['Remarks'] == 'Other Supplier']
        nn_result = nn_result[~nn_result[pred].isna()]
        nn_result.sort_values(by=['Sales Flag'], ascending=True, inplace=True)      ##
        nn_result.drop_duplicates(subset=['BU Code', 'BU Brand Name', 'IB Brand Name'], keep='first', inplace=True)       ##
        nn_result.reset_index(inplace=True, drop=True)
        nn_result.index = nn_result.index.set_names(['Prediction'])
        source = ['A-Brand', 'Luxury', 'Exclusive']

        with pd.ExcelWriter('./Output/Brand_Master_Bot.xlsx', engine='openpyxl') as writer:

            for sc in source:

                if sc == 'A-Brand':
                    pt_input = pd.read_excel(r'T:/Group File/Event/WIN MANAGEMENT/Brand Master/A-Brand/Brand Master - Abrand input - Jan 2018.xlsx', sheet_name='Brand Input', header=1)

                elif sc == 'Luxury':
                    pt_input = pd.read_excel(r'T:/Group File/Event/WIN MANAGEMENT/Brand Master/ILB/Brand Master_ILB.xlsx', sheet_name='Brand Input', header=1)
                    pt_input = pt_input.rename(columns={'IB Brand Name': 'Brand', 'BU Brand Name': 'BU brand name', 'IB Brand Name.1': 'Brand.1'})

                elif sc == 'Exclusive':
                    pt_input = pd.read_excel(r'T:/Group File/Event/WIN MANAGEMENT/Brand Master/EB/Brand Master - Exclusive Input.xlsx', sheet_name='Brand Input', header=1)
                    pt_input = pt_input.rename(columns={'IB Brand Name': 'Brand.1'})

                # pt_input = pt_input[pt_input['Updated Date'] >= one_year_before]
                pt_input = pt_input.drop_duplicates(subset=['Supplier', 'Brand', 'BU Display Name'], keep='last')
                pt_input = pt_input[pt_input['Status'] == 'Active']
                pt_input['Brand'] = pt_input['Brand'].str.upper()
                pt_input.reset_index(drop=True, inplace=True)

                pt_input = pt_input.iloc[:, 2:10]
                pt_input['BU Map'] = pt_input[['BU Display Name']].merge(bu_display_name, how='left').BU_Code

                # merge prediction
                # indexer = Index()
                # indexer.block(left_on='BU Map', right_on='BU Code')
                # pairs = indexer.index(pt_input, nn_result)
                # compare = Compare()
                #
                # algo = 'cosine'
                # compare.string('Brand', pred, method=algo, label='Prob')  ###
                # feature = compare.compute(pairs, pt_input, nn_result)
                #
                # feature.index.set_names(['Input', 'Prediction'], inplace=True)
                # feature.reset_index(level='Prediction', col_level=1, inplace=True)
                # idx = feature.groupby(['Input'])['Prob'].transform(max) == feature['Prob']
                # pt_input = pt_input.join(feature[idx])
                #
                # pt_input.loc[pt_input['Prob'] < 0.7, 'Prediction'] = -1
                # result = pt_input.merge(nn_result, on='Prediction', how='left')
                result = pt_input.merge(nn_result, left_on=['BU Map', 'Brand'], right_on=['BU Code', pred], how='left')

                result.loc[result['Mother Company (Prediction)'] == result['Supplier'], 'Supplier Match'] = 'Match'
                result.loc[result['Mother Company (Prediction)'] != result['Supplier'], 'Supplier Match'] = 'Unmatch'
                result = result.loc[:, ['Supplier', 'BU Display Name', 'Brand', 'Updated Date', 'BU Code',
                                        pred, 'Brand (Prob)', 'BU Brand Name', 'IB Brand Name',
                                        'Supplier Match', 'BU Supplier Code', 'BU Supplier Name', 'Mother Company (Prediction)',
                                        'Mother Company Code', 'Mother Company Name', 'Sales Flag']]
                result.dropna(subset=[pred], inplace=True)
                result.sort_values(by=['Brand (Prob)', pred], ascending=[False, True], inplace=True)
                result['Check Item Level'] = np.nan
                result.to_excel(writer, sheet_name=sc, index=False)
                have_sales = result[result['Sales Flag'] == 'Have Sales'].count()['Sales Flag']
                no_sales = result[result['Sales Flag'] == 'No Sales'].count()['Sales Flag']
                print(f'{sc}: {result.shape[0]} records found. | Have sales: {have_sales} | No sales: {no_sales}')

            print('Brand Master Bot Finished Running.\n')


    def item_level_bot(self):

        print('Item Level Bot Is Running......')
        source = ['A-Brand', 'Luxury', 'Exclusive']
        size_of_random_sample = 5

        with pd.ExcelWriter('./Output/Item_Level_Bot.xlsx', engine='openpyxl') as writer:

            for sc in source:

                result = pd.read_excel('./Output/Brand_Master_Bot.xlsx', sheet_name=sc)
                result = result[result['Check Item Level'].notna()].astype(str)
                bu = result['BU Code'].drop_duplicates().tolist()

                if not result.empty:
                    item_level=[[]]

                    for BU in bu:
                        temp = result[result['BU Code'] == BU]
                        brand = temp.groupby(['BU Code'])['BU Brand Name'].unique().transform(list).tolist()[0]
                        # size = len(brand) * size_of_random_sample
                        print(f'{BU} | Searching for:{brand}')
                        brand = ','.join("q'[" + item + "]'" for item in brand)
                        BU = "'"+BU+"'"
                        sql = SQL(BU=BU, Brand=brand, Size=size_of_random_sample)
                        item_level_sql = self.db.get_data(sql.item_level)
                        item_level_value = item_level_sql.values.tolist()
                        item_level.extend(item_level_value)

                    item_level_df = pd.DataFrame(columns=item_level_sql.columns, data=item_level).iloc[1:, :]
                    item_level_df.to_excel(writer, sheet_name=sc, index=False)

        print('Item Level Bot Finished Running.\n')


class Neural_Network_hierarchy_CNN:

    def __init__(self, db=None):

        self.db = db
        self.SQL = SQL()
        hierarchy = pd.read_excel('./training_source/Hierarchy.xlsx', dtype=str).astype(str)
        hierarchy.dropna(subset=['Local Hierarchy Desc1', 'Local Hierarchy Desc2', 'Local Hierarchy Desc3', 'Local Hierarchy Desc4'], inplace=True)

        hierarchy['Local Hierarchy (1-4)'] = hierarchy['Local Hierarchy Desc1'] + '|' + hierarchy['Local Hierarchy Desc2'] + '|' + hierarchy['Local Hierarchy Desc3'] + '|' + hierarchy[
            'Local Hierarchy Desc4']
        hierarchy['Global Hierarchy (1-4)'] = hierarchy['Global Hierarchy Desc1'] + '|' + hierarchy['Global Hierarchy Desc2'] + '|' + hierarchy['Global Hierarchy Desc3'] + '|' + hierarchy[
            'Global Hierarchy Desc4']
        hierarchy['Global Hierarchy Key'] = hierarchy['Global Hierarchy Code1'] + hierarchy['Global Hierarchy Code2'] + hierarchy['Global Hierarchy Code3'] + hierarchy['Global Hierarchy Code4']

        hierarchy = hierarchy[hierarchy['Local Hierarchy Desc1'] != 'UNMAP']
        hierarchy = hierarchy[hierarchy['Local Hierarchy Desc1'] != '-']
        self.hierarchy = hierarchy[hierarchy['Global Hierarchy Desc1'] != 'UNMAP']

        x = self.hierarchy.loc[:, ['BU Code', 'Local Hierarchy (1-4)']]
        y = self.hierarchy['Global Hierarchy (1-4)']

        self.hier_label_encoder = LabelEncoder()
        self.hier_label_encoder.fit(self.hierarchy['Global Hierarchy (1-4)'])

        self.x_train_hier, self.x_test_hier, self.y_train_hier, self.y_test_hier = train_test_split(x, y, test_size=0.33, shuffle=True, random_state=1)
        self.bu_Tokenizer = Tokenizer()
        self.bu_Tokenizer.fit_on_texts(x['BU Code'])
        self.x_Tokenizer = Tokenizer()
        self.x_Tokenizer.fit_on_texts(x['Local Hierarchy (1-4)'])
        self.maxlen = 22


    def train_hierarchy_CNN(self):

        print('Start Training Hierarchy Mapping Model......')
        no_of_label = len(self.hierarchy['Global Hierarchy (1-4)'].value_counts())
        x_train_bu = self.bu_Tokenizer.texts_to_matrix(self.x_train_hier['BU Code'])
        x_test_bu = self.bu_Tokenizer.texts_to_matrix(self.x_test_hier['BU Code'])
        x_train = self.x_Tokenizer.texts_to_sequences(self.x_train_hier['Local Hierarchy (1-4)'])
        x_train = pad_sequences(x_train, padding='post', truncating='post', maxlen=self.maxlen)  ##
        maxlen = x_train.shape[1]
        x_test = self.x_Tokenizer.texts_to_sequences(self.x_test_hier['Local Hierarchy (1-4)'])
        x_test = pad_sequences(x_test, padding='post', truncating='post', maxlen=self.maxlen)  ##

        y_train_hier_encode = self.hier_label_encoder.transform(self.y_train_hier)
        y_test_hier_encode = self.hier_label_encoder.transform(self.y_test_hier)

        V = len(self.x_Tokenizer.word_index)  # no. of vocab
        T = maxlen  # max. length of sequence (21)
        D = 64

        print(f'V: {V} | D: {D} | T: {T}')
        print(f'Training set: {self.x_train_hier.shape} | Test set: {self.x_test_hier.shape}')
        inputA = Input(batch_shape=(None, x_train_bu.shape[1]))
        inputB = Input(batch_shape=(None, x_train.shape[1]))

        one = Dense(256, activation='relu')(inputA)
        one = Dropout(0.5)(one)
        one = Dense(128, activation='relu')(one)
        one = Model(inputs=inputA, outputs=one)

        two = Embedding(input_dim=V + 1, output_dim=D, input_length=T)(inputB)
        two = Conv1D(filters=128, kernel_size=4, activation='relu', padding='causal')(two)
        two = Conv1D(filters=128, kernel_size=4, activation='relu', padding='causal')(two)
        two = BatchNormalization()(two)
        two = AveragePooling1D(4)(two)
        two = Flatten()(two)  ##
        two = Dropout(0.5)(two)
        two = Dense(512, activation='relu')(two)
        two = Model(inputs=inputB, outputs=two)

        combined = concatenate([one.output, two.output])
        combined = BatchNormalization()(combined)
        combined = Dense(no_of_label, activation='softmax')(combined)

        model = Model(inputs=[one.input, two.input], outputs=combined)
        model.summary()
        model.compile(loss=SparseCategoricalCrossentropy(),
                      optimizer=Adam(),
                      metrics=['accuracy'])
        model.fit([x_train_bu, x_train], y_train_hier_encode, epochs=8, batch_size=64,
                  callbacks=[LearningRateScheduler(scheduler2)],
                  validation_data=([x_test_bu, x_test], y_test_hier_encode))
        model.save('saved_model/NN_Hierarchy_Mapping_CNN.h5')
        model.evaluate([x_test_bu, x_test], y_test_hier_encode)
        print('Training Finished.\n')

        # y_pred = model.predict([x_test_bu, x_test])
        # y_prob = np.max(y_pred, axis=1)
        # y_pred = np.argmax(y_pred, axis=1)
        # y_pred = self.hier_label_encoder.inverse_transform(y_pred.tolist())
        # y_pred = pd.DataFrame({'Global Hierarchy (Prediction)': y_pred, 'Prob': y_prob})
        #
        # self.x_test_hier.reset_index(drop=True, inplace=True)
        # self.y_test_hier.reset_index(drop=True, inplace=True)
        # result = pd.concat([self.x_test_hier, y_pred, self.y_test_hier], axis=1)
        # result.loc[result['Global Hierarchy (Prediction)'] == result['Global Hierarchy (1-4)'], 'Accuracy'] = 'Correct'
        # result.loc[result['Global Hierarchy (Prediction)'] != result['Global Hierarchy (1-4)'], 'Accuracy'] = 'Incorrect'
        # result.to_excel('./hierarchy/NN_hierarchy_test_set_result.xlsx', index=False)


    def predict_unmap_hierarchy(self):

        print('Predicting 000 Unmap Hierarchy......')
        model = load_model('saved_model/NN_Hierarchy_Mapping_CNN.h5')
        # monthly_hier = pd.read_excel('./hierarchy/000 unmap hierarchy checking.xlsx', dtype=str)
        monthly_hier = self.db.get_data(self.SQL.unmap_hierarchy).astype(str)
        monthly_hier['Local Hierarchy (1-4)'] = monthly_hier['Local Hierarchy Desc1'] + '|' + monthly_hier['Local Hierarchy Desc2'] + '|' + monthly_hier['Local Hierarchy Desc3'] + '|' + \
                                                monthly_hier['Local Hierarchy Desc4']
        x = monthly_hier.loc[:, ['BU Code', 'Local Hierarchy (1-4)']]
        x_bu = self.bu_Tokenizer.texts_to_matrix(x['BU Code'])
        x_hier = self.x_Tokenizer.texts_to_sequences(x['Local Hierarchy (1-4)'])
        x_hier = pad_sequences(x_hier, padding='post', maxlen=self.maxlen)

        y_pred = model.predict([x_bu, x_hier])
        y_prob = np.max(y_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = self.hier_label_encoder.inverse_transform(y_pred.tolist())
        y_pred = pd.DataFrame({'Global Hierarchy (Prediction)': y_pred, 'Prob': y_prob})

        global_hier = self.hierarchy.loc[:, ['Global Hierarchy (1-4)', 'Global Hierarchy Key']]
        global_hier.drop_duplicates(inplace=True, keep='last')

        result = pd.concat([monthly_hier, y_pred], axis=1)
        result = result.merge(global_hier, how='left', left_on='Global Hierarchy (Prediction)', right_on='Global Hierarchy (1-4)')
        result.drop(result.iloc[:, [1, 2, 3, 4, -2]], inplace=True, axis=1)
        result.insert(result.shape[1] - 4, 'Global Hierarchy Key', result.pop('Global Hierarchy Key'))
        result.to_excel('./Output/000_hierarchy_prediction.xlsx', index=False)
        print('Prediction Finished.\n')


    def historical_hierarchy_check(self):

        model = load_model('saved_model/NN_Hierarchy_Mapping_CNN.h5')
        # monthly_hier = pd.read_excel('./training_source/Hierarchy.xlsx', dtype=str).astype(str)
        monthly_hier = self.db.get_data(self.SQL.Hierarchy)
        monthly_hier['Local Hierarchy (1-4)'] = monthly_hier['Local Hierarchy Desc1'] + '|' + monthly_hier['Local Hierarchy Desc2'] + '|' + monthly_hier['Local Hierarchy Desc3'] + '|' + \
                                                monthly_hier['Local Hierarchy Desc4']
        x = monthly_hier.loc[:, ['BU Code', 'Local Hierarchy (1-4)']]
        x_bu = self.bu_Tokenizer.texts_to_matrix(x['BU Code'])
        x_hier = self.x_Tokenizer.texts_to_sequences(x['Local Hierarchy (1-4)'])
        x_hier = pad_sequences(x_hier, padding='post', maxlen=self.maxlen)

        y_pred = model.predict([x_bu, x_hier])
        y_prob = np.max(y_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = self.hier_label_encoder.inverse_transform(y_pred.tolist())
        y_pred = pd.DataFrame({'Global Hierarchy (Prediction)': y_pred, 'Prob': y_prob})

        global_hier = self.hierarchy.loc[:, ['Global Hierarchy (1-4)', 'Global Hierarchy Key']]
        global_hier.drop_duplicates(inplace=True, keep='last')

        result = pd.concat([monthly_hier, y_pred], axis=1)
        result['Global Hierarchy (Validation)'] = monthly_hier['Global Hierarchy Desc1'] + '|' + monthly_hier['Global Hierarchy Desc2'] + '|' + monthly_hier['Global Hierarchy Desc3'] + '|' + \
                                                  monthly_hier['Global Hierarchy Desc4']
        result = result.merge(global_hier, how='left', left_on='Global Hierarchy (Prediction)', right_on='Global Hierarchy (1-4)')
        result.loc[result['Global Hierarchy (Prediction)'] == result['Global Hierarchy (Validation)'], 'Accuracy'] = 'Correct'
        result.loc[result['Global Hierarchy (Prediction)'] != result['Global Hierarchy (Validation)'], 'Accuracy'] = 'Incorrect'
        result = result.loc[:, ['BU Code', 'Local Hierarchy Code1', 'Local Hierarchy Desc1', 'Local Hierarchy Code2', 'Local Hierarchy Desc2', 'Local Hierarchy Code3', 'Local Hierarchy Desc3',
                                'Local Hierarchy Code4', 'Local Hierarchy Desc4', 'Global Hierarchy Key', 'Local Hierarchy (1-4)', 'Global Hierarchy (Prediction)', 'Prob',
                                'Global Hierarchy (Validation)', 'Accuracy']]

        result.to_excel('./Output/historical_hierarchy_prediction_result.xlsx', index=False)
        acc = result.query('Accuracy == "Correct"').shape[0] / result.shape[0]
        print(f'Accuracy: {acc * 100:.2f}%')


def bundle_check(self):
    model = load_model('saved_model/NN_Brand_Mapping_model_CNN.h5')
    bu_bundle = ['KV', 'ICI', 'HK', 'UK']
    with pd.ExcelWriter('brand_bundle_check/bundle_check_result.xlsx') as writer:
        for bu in bu_bundle:
            dataset = pd.read_excel('brand_bundle_check/data.xlsx', sheet_name=bu)
            dataset = dataset.astype(str)
            unique = dataset.groupby('BU Brand Name')['IB Brand Name'].unique().transform(len).rename('unique')
            dataset = dataset.merge(unique, on='BU Brand Name', how='left')
            x_real = dataset.loc[:, 'BU Brand Name']
            x_real_tfidf = self.brand_tokenizer.texts_to_matrix(x_real, mode='tfidf')
            y_pred = model.predict(x_real_tfidf)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = self.label_encoder.inverse_transform(y_pred.tolist())
            y_pred = pd.DataFrame(y_pred, columns=['IB Brand Name (Prediction)'])
            result = pd.concat([dataset, y_pred], axis=1)
            result = result[result['unique'] > 1]
            result = result.sort_values(by=['BU Brand Name'])
            result.to_excel(writer, sheet_name=bu, index=False)

def bundle_hierarchy_check():
    bu_bundle = ['KV', 'ICI', 'HK', 'UK']
    with pd.ExcelWriter('hierarchy/hierarchy_bundle_check_result.xlsx') as writer:
        for bu in bu_bundle:
            dataset = pd.read_excel('hierarchy/hierarchy_bundle.xlsx', sheet_name=bu)
            dataset.dropna(inplace=True)
            dataset.iloc[:, :8] = dataset.iloc[:, 0:8].astype(int)
            dataset = dataset.astype(str)
            dataset['Local Hierarchy Code(1-4)'] = dataset['Local Hierarchy Code1'] + '-' + dataset['Local Hierarchy Code2'] + '-' + dataset[
                'Local Hierarchy Code3'] + '-' + dataset['Local Hierarchy Code4']
            dataset['Global Hierarchy Code(1-4)'] = dataset['Global Hierarchy Code1'] + '-' + dataset['Global Hierarchy Code2'] + '-' + dataset[
                'Global Hierarchy Code3'] + '-' + dataset['Global Hierarchy Code4']
            unique = dataset.groupby('Local Hierarchy Code(1-4)')['Global Hierarchy Code(1-4)'].unique().transform(len).rename('unique')
            result = dataset.merge(unique, on='Local Hierarchy Code(1-4)', how='left')
            result = result[result['unique'] > 1]
            result = result.sort_values(by=['Local Hierarchy Code(1-4)'])
            result.to_excel(writer, sheet_name=bu, index=False)



