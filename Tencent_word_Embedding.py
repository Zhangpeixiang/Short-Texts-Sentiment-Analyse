### 2018/10/21/yuyuan

"""
Word_Num: 8824330
Word_Length: 200
Simlar_Method:Cosine Distance
Time Spend:around 45min
"""

import time
import numpy as np


# from tqdm import tqdm

def get_Word2Vec(n):
    # Using Iterable to Loading the Word2Vec data
    with open(r'D:/NLP/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8') as file:
        m = 0
        while m < n:
            yield file.readline().split()
            m += 1


def search_Word(Need_Word_dict):
    # To get the Word2Vec in which words u want
    for i in get_Word2Vec(8824330):
        if i[0] in Need_Word_dict:
            Need_Word_dict[i[0]] = i[1::]
            break

    return Need_Word_dict


def transfer_type(Need_Word_dict):
    vec_list = []
    for key in Need_Word_dict:
        for i in Need_Word_dict[key]:
            vec_list.append(float(i))

    return vec_list


def transfer_type2(vector):
    vec2 = []
    for i in vector[1::]:
        try:
            vec2.append(float(i))
        except ValueError:
            print("ValueError: ===========>", vector)
    return vec2


def Cos_Distance(vec_1_list, vec_2_list):
    #     print (vec_2_list)
    vector_a = np.mat(vec_1_list)
    vector_b = np.mat(vec_2_list)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def Get_Sim_Word(Need_Word_dict):
    sim_list = []
    vec_1_list = transfer_type(Need_Word_dict)
    for i in get_Word2Vec(8824330):
        if i[0] != '8824330' and len(i) == 201:
            vec_2_list = transfer_type2(i)
            cos_dis = Cos_Distance(vec_1_list, vec_2_list)
            if cos_dis >= 0.7:
                sim_couple = [i[0], cos_dis]
                sim_list.append(sim_couple)
                if len(sim_list) > 100:
                    ###看来远远的不止1W，阈值设定的低了
                    break
    sim_list.sort(key=lambda x: x[1], reverse=True)

    return sim_list


if __name__ == "__main__":
    start = time.time()
    ###modify the word
    word_dict = {'，': 0}
    Need_Word_dict = search_Word(word_dict)
    t1 = time.time()
    print('Word has been found: ', t1 - start)
    # Sim_Word = Get_Sim_Word(Need_Word_dict)
    end = time.time()
    print('Time_Spending : ', end - t1, end - start)