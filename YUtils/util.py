import pandas as pd

def Add_list_colum(l,a):
    b = pd.DataFrame(l).T
    b.columns = a.columns
    c = pd.concat([a,b],ignore_index=False)
    return c


def getkeylist(sample_dic):
    key_list=[]
    value_list=[]
    for key, value in sample_dic.items():
        key_list.append(key)
        value_list.append(value)

    return key_list