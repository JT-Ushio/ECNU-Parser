#encoding: utf-8
from data_loader import *
import pandas as pd
from os import listdir

class tree_banks_loader(object):
    def __init__(self, filename, table_filename, train = True):
        self.language_table = pd.read_csv(table_filename)
        self.language_table.set_index("language",inplace=True)
        self.filename = filename#数据集的文件路径
        filelist = listdir(filename)#共计82个树库
        self.data = {}
        for example in filelist:
            self.data[example]={}
            language = example[example.index("_")+1:example.rindex("-")]
            self.data[example]["language"] = language
            self.data[example]["family"] = self.language_table.loc[language]["family"]
            self.data[example]["genus"] = self.language_table.loc[language]["genus"]
            #处理每个树库文件夹中的内容
            filestr = filename + "/" + example
            flist = listdir(filestr)
            flag = 0#标记是否有验证集
            length = 0
            flag2 = 0
            for dev_file in flist:
                if dev_file.find("dev.conllu") != -1:
                    flag = 1
                    break
            for train_file in flist:
                if train_file.find("train.conllu") != -1:
                    flag2 = 1
                    break
            if flag2 == 0:#对于不存在训练集的数据，dataset为空
                self.data[example]['dataset'] = []
                continue
            if train or (train == False and flag == 0):
                MAXN_CHAR = 30
                TRAIN_FILE = filestr + "/" + train_file
                self.data[example]['dataset'], _ = build_dataset(TRAIN_FILE, MAXN_CHAR, nonproj=True, train=True)
                #若不存在验证集，训练集只取前70%
                length = int(0.7 * len(self.data[example]['dataset']))
                if train and flag == 0:
                    self.data[example]['dataset'] = self.data[example]['dataset'][:length]
                elif train == False and flag == 0:
                    self.data[example]['dataset'] = self.data[example]['dataset'][length:]
            else:
                if flag:
                    MAXN_CHAR = 30
                    DEV_FILE = filestr + "/" + dev_file
                    self.data[example]['dataset'], _ = build_dataset(DEV_FILE, MAXN_CHAR,train=False)
                return



    def get_treebanks_by_name(self, name):
        return self.data[name]

    def get_treebanks_by_language(self, language):
        dict = {}
        for key in self.data:
            if self.data[key]['language'] == language:
                dict[key] = self.data[key]
        return dict

    def get_treebanks_by_family(self, family):
        dict = {}
        for key in self.data:
            if self.data[key]['family'] == family:
                dict[key] = self.data[key]
        return dict

    def get_treebanks_by_genus(self, genus):
        dict = {}
        for key in self.data:
            if self.data[key]['genus'] == genus:
                dict[key] = self.data[key]
        return dict

    def get_all_treebanks(self):
        return self.data

if __name__ == '__main__':
    filename = 'C:/Users/Fang/Desktop/conll/ud-treebanks-v2.2'#树库的文件路径
    table_filename = 'C:/Users/Fang/Desktop/conll/language_table.csv'#语言表的文件路径
    loader = tree_banks_loader(filename, table_filename, True)
    data = loader.get_treebanks_by_name("UD_Afrikaans-AfriBooms")
    data2 = loader.get_treebanks_by_language("Afrikaans")
    data3 = loader.get_treebanks_by_family("Indo-European")
    data4 = loader.get_treebanks_by_genus("Germanic")
    data5 = loader.get_all_treebanks()
    print("OK")


