clc; clear; close all;
addpath(genpath('.')); % 添加当前目录及其子目录到搜索路径
DBDIR = './Dataset/'; % 数据集的存储路径
i = 1;

% 注释掉不需要的数据集，按需求选择数据集
DataName{i} = '3sources'; i = i + 1;
%DataName{i} = 'ORL'; i = i + 1;
%DataName{i} = 'Caltech101-7'; i = i + 1;
%DataName{i} = 'Caltech101-20'; i = i + 1; % 启用Caltech101-20数据集
%DataName{i} = 'HW2sources'; i = i + 1;
%load('Caltech101-20.mat'); % 加载Caltech101-20数据集
%whos % 查看当前工作区变量信息


dbNum = length(DataName); % 数据集数量
maxIter = 10; % 最大迭代次数
bnorm = 1; % 是否对数据进行归一化
knn = 20; % 构造SIG时的邻居数量，通常设置为15效果较好
lambda = [1 1000 10]; % lamda参数，其中lambda1通常固定为1



testLamda = 0; % 如果想在其他数据集上运行代码，设置为1以获得最佳结果
for dbIndex = 1:dbNum
    clear X gt; % 清除变量
    dbnamePre = DataName{dbIndex}; % 当前数据集名称
    dbfilename = sprintf('%s%s.mat', DBDIR, dbnamePre); % 构造数据集文件路径
    load(dbfilename); % 加载数据集
    
    % 根据不同的数据集进行数据处理
    if strcmp(DataName{1}, '3sources')
        gt = truelabel{2}'; % 标签
        X = data; % 数据
    elseif strcmp(DataName{1}, 'Caltech101-7')
        for i = 1:6
            X{i} = X{i}'; % 转置数据
        end
        gt = Y; % 标签
    elseif strcmp(DataName{1}, 'Caltech101-20')
        for i = 1:6
            X{i} = X{i}'; % 转置数据
        end
        gt = Y; % 标签
       elseif strcmp(DataName{1}, 'ORL')
        for i = 1:4
             X{i} = X{i}'; % 转置数据
         end
         gt = Y; % 标签
    elseif strcmp(DataName{1}, 'HW')
        gt = truelabel{1}; % 标签
        X = data; % 数据
    elseif strcmp(DataName{1}, 'HW2sources')
        gt = truelabel{1}; % 标签
        X = data; % 数据
    elseif strcmp(DataName{1}, 'MSRCv1')
         for i = 1:5
             X{i} = X{i}'; % 转置数据
         end
         gt = Y; % 标签
    end

    gnd = gt; % 真实标签
    la2 = [10000 1000 100 10 1 0.1 0.01 0.001]; % lambda2参数的可选值
    la3 = [0.001 0.01 0.1 1 10 100 1000 10000]; % lambda3参数的可选值

    % 针对特定数据集调整lambda参数
    if contains(dbnamePre, 'Caltech101-20')
        lambda = [1 1000 0.1]; % 为Caltech101-20数据集设置的最优lambda参数
    elseif contains(dbnamePre, 'BBC') && ~contains(dbnamePre, 'BBCSport')
        lambda = [1 1 0.01]; % 保留以防需要使用BBC数据集
    end

    fprintf("正在运行 [%s]...\n", dbnamePre); % 打印当前运行的数据集名称

    % 如果需要测试不同的lambda值
    if (testLamda == 1)
        maxIter = 10; % 调整最大迭代次数
        if (dbIndex > 4)
            maxIter = 5; % 对于特定的数据集减少迭代次数
        end

        for i = 1:length(la2)
            for j = 1:length(la3)
                lambda(2) = la2(i);
                lambda(3) = la3(j);
                if (lambda(2) <= lambda(3))  % 当lambda(2) <= lambda(3)时，结果不好，忽略此情况
                    continue;
                end
                result = UMSR(X, bnorm, knn, gnd, maxIter, lambda); % 调用算法函数
                printResult(result, dbnamePre, lambda); % 打印结果
            end
        end
    else
        tic; % 开始计时
        result = UMSR(X, bnorm, knn, gnd, maxIter, lambda); % 调用算法函数
        printResult(result, dbnamePre, lambda); % 打印结果
    end
end
fprintf('\n 完成... \n'); % 运行完成

% 打印结果的函数
function printResult(result, dbnamePre, lamda)
str = sprintf('[%s 迭代:%d][准确率 %.2f] [NMI %.2f] [ARI %.2f] [F-score %.2f] [Purity %.2f] [Precision %.2f] [Recall %.2f][lambda2:%.3f lambda3:%.3f] [执行时间:%.2f] %s ',...
    dbnamePre, result(8), result(1) * 100,  result(2) * 100, result(3) * 100, result(4) * 100, result(5) * 100, result(6) * 100, result(7) * 100, lamda(2), lamda(3), toc, GetTimeStrForLog());
fprintf('%s\n',str); % 打印格式化后的结果字符串
end
