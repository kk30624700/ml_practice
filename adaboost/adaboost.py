'''
Created on 2018年8月23日

@author: admin
'''
import numpy as np
def laod_sim_data():
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels

def load_data_set(file_name):
    num_feat = len(open(file_name).readline().split('\t'))
    data_arr = []
    label_arr = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat-1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))
    return np.matrix(data_arr), label_arr

def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    return ret_array

def build_stump(data_arr, class_labels, D):
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    
    m, n  = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    
    min_err = np.inf
    
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps 
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_err = D.T * err_arr
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_err, best_class_est

def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1))/m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        agg_class_est += alpha * class_est
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est

def ada_classify(data_to_class, classifier_arr):
    data_mat = np.mat(data_to_class)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)
    return np.sign(agg_class_est)

def plot_roc(pred_strengths, class_labels):
    """
    (打印ROC曲线，并计算AUC的面积大小)
    :param pred_strengths: 最终预测结果的权重值
    :param class_labels: 原始数据的分类结果集
    :return: 
    """
    import matplotlib.pyplot as plt
    # variable to calculate AUC
    y_sum = 0.0
    # 对正样本的进行求和
    num_pos_class = np.sum(np.array(class_labels) == 1.0)
    # 正样本的概率
    y_step = 1 / float(num_pos_class)
    # 负样本的概率
    x_step = 1 / float(len(class_labels) - num_pos_class)
    # np.argsort函数返回的是数组值从小到大的索引值
    # get sorted index, it's reverse
    sorted_indicies = pred_strengths.argsort()
    # 测试结果是否是从小到大排列
    # 可以选择打印看一下
    # 开始创建模版对象
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # cursor光标值
    cur = (1.0, 1.0)
    # loop through all the values, drawing a line segment at each point
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # draw line from cur to (cur[0]-delX, cur[1]-delY)
        # 画点连线 (x1, x2, y1, y2)
        # print cur[0], cur[0]-delX, cur[1], cur[1]-delY
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    # 画对角的虚线线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    # 设置画图的范围区间 (x1, x2, y1, y2)
    ax.axis([0, 1, 0, 1])
    plt.show()
    '''
    参考说明：http://blog.csdn.net/wenyusuran/article/details/39056013
    为了计算 AUC ，我们需要对多个小矩形的面积进行累加。
    这些小矩形的宽度是x_step，因此可以先对所有矩形的高度进行累加，最后再乘以x_step得到其总面积。
    所有高度的和(y_sum)随着x轴的每次移动而渐次增加。
    '''
    print("the Area Under the Curve is: ", y_sum * x_step)


def test():
    # D = np.mat(np.ones((5, 1)) / 5)
    # data_mat, class_labels = load_sim_data()
    # print(data_mat.shape)
    # result = build_stump(data_mat, class_labels, D)
    # print(result)
    # classifier_array, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 9)
    # print(classifier_array, agg_class_est)
    data_mat, class_labels = load_data_set('./horseColicTraining2.txt')
    print(data_mat.shape, len(class_labels))
    weak_class_arr, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 40)
    print(weak_class_arr, '\n-----\n', agg_class_est.T)
    plot_roc(agg_class_est, class_labels)
    data_arr_test, label_arr_test = load_data_set("./horseColicTest2.txt")
    m = np.shape(data_arr_test)[0]
    predicting10 = ada_classify(data_arr_test, weak_class_arr)
    err_arr = np.mat(np.ones((m, 1)))
    # 测试：计算总样本数，错误样本数，错误率
    print(m,
          err_arr[predicting10 != np.mat(label_arr_test).T].sum(),
          err_arr[predicting10 != np.mat(label_arr_test).T].sum() / m
          )


if __name__ == '__main__':
    test()
