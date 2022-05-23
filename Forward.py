import numpy as np

input_user = [0.9, 0.1, 0.8]
# cmd = input("형식에 맞춰 input값을 입력하세요 ex) Input [[0.9],[0.1],[0.8]]\n")
# input_user = []
# input_user.append(float(cmd[8:11]))
# input_user.append(float(cmd[14:17]))
# input_user.append(float(cmd[20:23]))

input_hidden = ([0.9, 0.3, 0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]) # 첫 번째 layer에 있는 각 노드들의 weight값
hidden_output = ([0.3,0.7,0.5],[0.6,0.5,0.2],[0.8, 0.1,0.9]) # 두 번째 layer에 있는 각 노드들의 weight값

def sigmoid(x):
    return 1/(1+np.exp(-x))

def matrix_mul(input, input_hidden):
    b = np.array(input)
    a = np.array(input_hidden)
    output_hidden = np.dot(a,b)
    print("matrix_output:",output_hidden)
    output_sig = []
    for i in output_hidden:
        output_sig.append(sigmoid(i))
    return output_sig

target = [0.6, 0.8, 0.5]

if __name__ == '__main__':
    #i j k
    second_output = matrix_mul(input_user, input_hidden) # output j
    print("sig output:",second_output)
    final_output = matrix_mul(second_output, hidden_output) # output k
    print("sig output:",final_output)
    actual = []
    for i in final_output:
        string = str(i)
        num = ''
        for j in range(5):
            num += string[j]
        actual.append(float(num))

    # target = [0.6, 0.8, 0.5]
    # actual = [0.726, 0.708, 0.778]
    #print([target[i]-actual[i] for i in range (len(target))])
    #print([x-y for x,y in zip(target, actual)])

    test_w2 = ([2.0, 3.0], [1.0, 4.0])
    test_e = [0.8, 0.5]
    test_w1 = ([3.0, 2.0], [1.0, 7.0])

    ## 1. (tk - ok)
    e_output = ([target[i] - actual[i] for i in range(len(target))])

    e1 = test_e[0]
    ## 2. wjk  oj
    oj = [0.4, 0.5]
    oj = np.array(oj)
    wjk = test_w2[0]
    wjk = np.array(wjk)
    sum = np.dot(oj,wjk)

    ## 3. sigmoid
    sig =round(sigmoid(sum),3)

    ##4. oj1
    oj1 = oj[0]

    ## result

    #zip
    order = "123"
    animals = ['lion', 'cat', 'dog']
    fruits = ['mandarin', 'strawberry', 'apple']
    zip_list = zip(order, animals, fruits)
    print(list(zip_list))
    zip_dict = zip(order, animals)
    print(dict(zip_dict))

    data = {}

    ret = data.setdefault('a', 0)    # key로 'a'를 추가학 value 0을 설정함, setdefault는 현재 value 값을 리턴
    print(ret, data)

    ret = data.setdefault('a', 1)   # 이미 key가 있는 경우 setdefault 현재 value 값을 리턴
    print(ret, data)