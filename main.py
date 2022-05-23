import numpy as np

input_user = [0.9, 0.1, 0.8]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def matrix_mul(input, input_hidden):
    b = np.array(input)
    a = np.array(input_hidden)
    output_hidden = np.dot(a, b)
    output_sig = []
    for i in output_hidden:
        output_sig.append(sigmoid(i))
    return output_sig


def get_error(target, actual):
    e_output = ([target[i] - actual[i] for i in range(len(target))])
    return e_output


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


# def get_new_weight(weight_jk, diff):
#     weight_jk = np.array(weight_jk)
#     new_weights = []
#     for i in range(3):
#         temp = []
#         for j in range(3):
#             w = weight_jk[i][j]  # 갱신할 weight하나
#             #new_w = round(w - (0.5 * diff[i]), 4)
#             new_w = w - (0.5 * diff[i])
#             temp.append(new_w)
#         new_weights.append(temp)
#     new_weights = np.array(new_weights)
#     return new_weights

    # w00 = test_w2[0][0]
    # new_w00 = round(w00 - (0.1 * diff), 4)


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


# def get_diff(error, output, output_j):
#     diff = []
#     for i in range(3):
#         # E - a20
#         err = error[i]
#         # a20 - z20
#         temp = output[i] * (1 - output[i])
#         # z20 - w1.10
#         # output_j[i]
#         result = -err * temp * output_j[i]
#         diff.append(result)
#     return diff


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


def forward(layer1_weights, layer2_weights):
    second_output = matrix_mul(input_user, layer1_weights)
    final_output = matrix_mul(second_output, layer2_weights)
    actual = []
    for i in final_output:
        actual.append(round(i,8))
    return second_output, actual


def backward(target, actual, second_output, layer2_weights, input_user, layer1_weights):
    e_output = get_error(target, actual)
    #diff = get_diff(e_output, actual, second_output)
    diff = get_diff(e_output, layer2_weights, second_output)
    new_layer2_weights = get_new_weight(layer2_weights, diff)
    diff_layer1 = get_diff_layer1(e_output, actual, layer2_weights, second_output, input_user)
    new_layer1_weights = get_new_weight(layer1_weights, diff_layer1)
    return new_layer1_weights, new_layer2_weights


target = [0.6, 0.8, 0.5]
count = 0
if __name__ == '__main__':
    layer1_weights = ([0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6])  # 첫 번째 layer에 있는 각 노드들의 weight값
    layer2_weights = ([0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9])  # 두 번째 layer에 있는 각 노드들의 weight값

    while(count <1000):
        second_output, actual = forward(layer1_weights,layer2_weights)
        print(actual)
        new_l1, new_l2 = backward(target, actual, second_output, layer2_weights, input_user, layer1_weights)
        layer1_weights = new_l1
        layer2_weights = new_l2
        count +=1
    print("after Training output")
    print("[["+str(actual[0])+"]")
    print(" ["+str(actual[1])+"]")
    print(" ["+str(actual[2])+"]]")


