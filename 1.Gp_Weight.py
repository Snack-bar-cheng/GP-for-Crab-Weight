# python packages
import random
import  time
import itertools
import operator
import evalGP_main as evalGP
from evalGP_main import hof_generation_list,fitness_generation_best
from  Function_Gp import  Generation_test,Plot_tree,PFI_feature
# only for strongly typed GP
import gp_restrict
import numpy as np
import pandas as pd
# deap package
from deap import base, creator, tools, gp
import pygraphviz as pgv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import Counter
import warnings
# warnings.filterwarnings("ignore")

randomSeeds=30


path_train =r"F:\工作 pycharm\4.回归问题——第二篇sci\3.审稿后重做实验\螃蟹全重数据集和结果记录\CrabWeightPrediction_train.csv"
path_test =r"F:\工作 pycharm\4.回归问题——第二篇sci\3.审稿后重做实验\螃蟹全重数据集和结果记录\CrabWeightPrediction_test.csv"

Gp_Tree_save_place_hof = r"F:\工作 pycharm\4.回归问题——第二篇sci\3.审稿后重做实验\螃蟹全重数据集和结果记录\hof_best_tree_4"

train_data = np.loadtxt(path_train, delimiter=',', skiprows=1)
test_data = np.loadtxt(path_test, delimiter=',', skiprows=1)


X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]


print("Train:",X_train.shape)
print("Test:",X_test.shape)

# parameters:
population = 1024
generation = 50
cxProb = 0.7
mutProb = 0.2
elitismProb = 0.02
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 3
maxDepth = 4


# Define new functions
def protectedDiv(left, right):
    if right != 0:
        return left / right
    else:
        return 1



pset = gp.PrimitiveSetTyped('MAIN', itertools.repeat(float, X_train.shape[1]),float)

data = pd.read_csv(path_test)
# 获取特征名，排除最后一列（预测变量）
feature_names = data.columns[:-1]
# print(feature_names)
feature_Name = []
# 动态重命名变量名
for i, name in enumerate(feature_names):
    # name ="Ln_{}".format(name)
    pset.renameArguments(**{f'ARG{i}': name})
    feature_Name.append(name)

feature_names = feature_Name

pset.addPrimitive(operator.add,[float, float], float,name='Add')
pset.addPrimitive(operator.sub,[float, float], float,name='Sub')
pset.addPrimitive(operator.mul,[float, float], float,name='Mul')
pset.addPrimitive(protectedDiv,[float, float], float,name='Div')



pset.addEphemeralConstant("rand", lambda: random.uniform(-10, 10), float)

#fitnesse evaluaiton
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)



def evalTrain(individual):
    # print(individual)
    func = toolbox.compile(expr=individual)
    # print(func)

    y_pred = []
    # 计算预测值
    for i in range(0, len(y_train)):
         y_pred_number = func(*X_train[i, :])
         y_pred.append(y_pred_number)

    # 计算评估指标
    r_squared = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)

    return mse,


# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof



def evalTest(individual):
    func = toolbox.compile(expr=individual)

    y_pred = []
    # 计算预测值
    for i in range(0, len(y_test)):
        y_pred_number = func(*X_test[i, :])
        y_pred.append(y_pred_number)

    # 计算评估指标
    r_squared = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return r_squared,  mse, rmse, mae


if __name__ == "__main__":

    # 训练部分
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime
    # 测试部分
    endTime2 = time.process_time()
    testTime = endTime2 - endTime

    ###打印记录/实验结果
    # 输出训练模型的时间/测试时间
    print("TrainTime:", trainTime)
    print("TestTime:", testTime)


    # 输出GP每一代最好的个体在训练集上的学习情况
    print("+" * 50 + "Training set: 每一代最好个体 R²(如下)" + "+" * 50)
    print(fitness_generation_best)
    print("+" * 50 + "Training set: 每一代最好个体 R²(如上)" + "+" * 50)
    print()


    # 输出GP每一代最好的个体在测试集上的学习情况
    gene_max_index = Generation_test(hof_generation_list, evalTest, randomSeeds)
    print()


    # 输出Hof0的个体表示以及测试集上的学习情况
    print("+" * 145)
    R_squared, Mse, Rmse, Mae = evalTest(hof[0])
    print('Best individual:', hof[0])
    print(f"R²:{R_squared}，MSE:{Mse}, RMSE:{Rmse}, MAE:{Mae}")
    print("+" * 145)
    print()

    # 输出hof中个体所选择的各个特征的数量(如下)/将hof可视化存入本地 //
    ##统计特征数量的时候，我们的目标是好的模型在train和test都好的模型，这个地方要限制，而不是直接拿HOF中的
    Plot_tree(hof[0], randomSeeds, Gp_Tree_save_place_hof, feature_names)
    print()

    PFI_feature(X_test, y_test, hof[0],pset,1)  # 计算PFI计算特征重要性
