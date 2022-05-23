import numpy as np

#sigmoid 수행 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# forward중 행렬곱 수행하고 sigmoid에 넣어 output을 생성하는 함수
def matrix_mul(input, input_hidden):
    b = np.array(input)
    a = np.array(input_hidden)
    output_hidden = np.dot(a, b)
    output_sig = []
    for i in output_hidden:
        output_sig.append(sigmoid(i))
    return output_sig


# target과 actual 값을 이용해 error값 계산, 각 노드의 error를 리스트로 반환
def get_error(target, actual):
    e_output = ([target[i] - actual[i] for i in range(len(target))])
    return e_output


# 두 번째 layer에 있는 weight들의 가중치(기울기)를 계산해서 각 루트별로 가중치 리스트 반환
def get_diff(error, weights_jk, output_j):
    weights_jk = np.array(weights_jk)
    diff = []
    for i in range(3):
        err = error[i]
        temp = np.dot(output_j, weights_jk[i])
        sig = sigmoid(temp)
        d = -err * sig * (1 - sig) * output_j[i]
        diff.append(d)
    return diff

# weight 리스트를 가중치에 따라 갱신해서 new_weight 리스트 반환
def get_new_weight(weight_jk, diff):
    weight_jk = np.array(weight_jk)
    new_weights = []
    for i in range(3):
        temp = []
        for j in range(3):
            w = float(weight_jk[i][j])
            new_w = w - (0.1 * float(diff[i]))
            temp.append(new_w)
        new_weights.append(temp)
    new_weights = np.array(new_weights)
    return new_weights

# 첫 번째 layer에 있는 weight들의 가중치(기울기)를 계산해서 각 루트별로 가중치 리스트 반환
def get_diff_layer1(error, output, layer2_weights, output_j, input_user):
    diff = []
    three_nodes = 0
    for i in range(3):
        for j in range(3):
            # Ei - a10
            err = error[i]
            temp = output[i] * (1 - output[i])
            next_w = layer2_weights[i][j]
            three_nodes += -err * temp * next_w
        # a10 - z10
        sig = output_j[i] * (1 - output_j[i])

        # z10 - w0.10 = x1? input?
        i = input_user[i]
        diff.append(three_nodes * sig * i)
    return diff

# forward 함수, hidden layer2의 output과 actual를 반환
def forward(layer1_weights, layer2_weights):
    second_output = matrix_mul(input_user, layer1_weights)
    final_output = matrix_mul(second_output, layer2_weights)
    actual = []
    for i in final_output:
        actual.append(round(i,8))
    return second_output, actual

# backpropagation 함수, 새로운 weight 리스트들을 반환
def backward(target, actual, second_output, layer2_weights, input_user, layer1_weights):
    e_output = get_error(target, actual)
    diff = get_diff(e_output, layer2_weights, second_output)
    new_layer2_weights = get_new_weight(layer2_weights, diff)
    diff_layer1 = get_diff_layer1(e_output, actual, layer2_weights, second_output, input_user)
    new_layer1_weights = get_new_weight(layer1_weights, diff_layer1)
    return new_layer1_weights, new_layer2_weights

input_user = [0.9, 0.1, 0.8] # 처음 입력값
target = [0.6, 0.8, 0.5] # 타겟
count = 0 # 아래 while문을 나가기위한 count값

if __name__ == '__main__':
    layer1_weights = ([0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6])  # 첫 번째 layer에 있는 각 노드들의 weight값
    layer2_weights = ([0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9])  # 두 번째 layer에 있는 각 노드들의 weight값

    # backpropagation을 1000번 수행
    while(count <1000):
        second_output, actual = forward(layer1_weights,layer2_weights)
        new_l1, new_l2 = backward(target, actual, second_output, layer2_weights, input_user, layer1_weights)
        layer1_weights = new_l1
        layer2_weights = new_l2
        count +=1

    # backpropagation 1000번 후의 output 출력
    print("after Training output")
    print("[["+str(actual[0])+"]")
    print(" ["+str(actual[1])+"]")
    print(" ["+str(actual[2])+"]]")


