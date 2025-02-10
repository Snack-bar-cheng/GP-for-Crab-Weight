############################################################################################################
#                 可视化图区域
############################################################################################################
# 1.可视化GP_Tree: Plot_Tree ；
# 2.统计所选择的Feature 数量：用于与其他方法对比特征的重要性 ；
############################################################################################################

import pygraphviz as pgv
from collections import Counter
from deap import gp

def Plot_tree(tree,randomSeeds,tree_save_fileplace,feature_names,generation_i=0):

    feature_count_list = []

    generation_i +=1
        # 生成绘制GP树所需的元素
    nodes, edges, labels = gp.graph(tree)

    # 打印最好的个体所使用的特征和数字
    value_counts = Counter(labels.values())
    count_feature = [value_counts.get(name, 0) for name in feature_names]

    feature_count_list.append(count_feature)

    for node, label in labels.items():
        if isinstance(tree[node], gp.Terminal):
            if isinstance(tree[node].value, float):
                tree[node].value = round(tree[node].value, 4)
                labels[node] = tree[node].value


        # 创建空的有向图对象
    graph = pgv.AGraph(directed=True)

         # 添加节点和边
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

        # 将标签设置为节点的标签，并指定节点的颜色
    for node, label in labels.items():
        if isinstance(tree[node], gp.Terminal):
            node_obj = graph.get_node(node)
            node_obj.attr['label'] = label
            node_obj.attr['style'] = 'filled'
            node_obj.attr['fillcolor'] = '#CDEB8B'  # 设置终端节点的填充颜色

        else:
            node_obj = graph.get_node(node)
            node_obj.attr['label'] = label
            node_obj.attr['style'] = 'filled'
            node_obj.attr['fillcolor'] = '#FFCC99'  # 设置非终端节点的填充颜色


        # 使用pygraphviz的绘图功能绘制GP树
    graph.layout(prog='dot')
    graph.draw(tree_save_fileplace +"\\" +"randomSeeds = {}时,hof {}.pdf".format(randomSeeds, generation_i))

    column_sums = [sum(col) for col in zip(*feature_count_list)]
    print("randomSeeds = {}时,Hof 已可视化存入；该个体的统计特征总和为：{},".format(randomSeeds,column_sums))


############################################################################################################
#                   特征重要性
############################################################################################################
# 1.Feature_PFI: 随机打乱某一列特征，看MSE；
############################################################################################################

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from deap import gp



def PFI_feature(X_test, y_test, individual, pset, randomSeeds=None):

    # 设置随机种子确保结果可重复
    if randomSeeds is not None:
        np.random.seed(randomSeeds)

    func = gp.compile(expr=individual,pset=pset)

    # 计算原始预测值
    Yorig_pred = [func(*X_test[i, :]) for i in range(len(y_test))]

    # 计算原始的MSE
    MSEorig = mean_squared_error(y_test, Yorig_pred)

    # 初始化特征重要性数组
    FI = np.zeros(X_test.shape[1])

    # 对每个特征进行置乱并计算PFI
    for j in range(X_test.shape[1]):  # 遍历每个特征

        # 置乱特征数据
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, j])  # 打乱某个特征列

        # 计算置乱后的预测值
        Ypmt_pred = [func(*X_test_permuted[i, :]) for i in range(len(y_test))]

        # 计算置乱后的MSE
        MSEpmt = mean_squared_error(y_test, Ypmt_pred)

        # 计算特征重要性
        FI[j] = MSEpmt - MSEorig

    FI = list(FI)

    # 输出特征重要性
    print("randomSeeds = {} 时 特征重要性:{}".format(randomSeeds,FI))

    # 返回特征重要性
    return FI



############################################################################################################
#                 GP学习指标区域
############################################################################################################
# 1.Generation_test:①GP将学习多代数的模型，取每一代中最好的一代在测试集上得到指标；绘图来查看GP是否变得更好 ；
#                   ②以同样的操作在训练集上得到的指标，对比是否过拟合或欠拟合；
############################################################################################################
def Generation_test(hof_list,Test_Function,randomSeeds):
    # 测试每一代的最好的个体
    gen_count = 0
    # 设定几个评估指标的列表---用来存储每一次（共30次）运行的结果50次迭代
    r_squared_tf = []
    mse_tf = []
    rmse_tf = []
    mae_tf = []

    for hof in hof_list:
        r, mse, rmse, mae = Test_Function(hof)

        # 打印评估指标
        # r = round(r, 4)
        # print("R²:", r)
        r_squared_tf.append(r)

        # mse = round(mse, 4)
        # print("MSE:", mse_r)
        mse_tf.append(mse)

        # rmse = round(rmse, 4)
        # print("RMSE:", rmse_r)
        rmse_tf.append(rmse)

        # mae = round(mae, 4)
        # print("MAE:", mae_r)
        mae_tf.append(mae)


    print("randomSeeds ={}时  R²最好：{}； ".format(randomSeeds,max(r_squared_tf)))
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 R² (如下)" + "+" * 50)
    print(r_squared_tf)
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 R² (如上)" + "+" * 50)
    print()

    print("randomSeeds ={}时  MSE;".format(randomSeeds))
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 MSE (如下)" + "+" * 50)
    print(mse_tf)
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 MSE (如上)" + "+" * 50)
    print()

    print("randomSeeds ={}时  RMSE;".format(randomSeeds))
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 RMSE (如下)" + "+" * 50)
    print(rmse_tf)
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 RMSE (如上)" + "+" * 50)
    print()

    print("randomSeeds ={}时  MAE;".format(randomSeeds))
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 MAE (如下)" + "+" * 50)
    print(mae_tf)
    print("+" * 50 + "输出每一代最好个体在测试集上的表现 MAE (如上)" + "+" * 50)
    print()

    # print("R², Mse, Rmse, Mae")
    # Evaluation_indicators = [r_squared_tf, mse_tf, rmse_tf, mae_tf]
    # print("+" * 50 +"输出每一代最好个体在测试集上的表现 合集 (如下)" + "+" * 50)
    # print(Evaluation_indicators)
    # print("+" * 50 + "输出每一代最好个体在测试集上的表现 合集 (如上)" + "+" * 50)

    # 使用max函数找到最大值
    max_value = max(r_squared_tf)
    # 使用index方法找到最大值的索引
    max_index = r_squared_tf.index(max_value)

    return  max_index


