import pandas as pd
import re
import cpca
import numpy as np
import matplotlib.pyplot as plt
# 设置中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

index = 1

def clean():
    data = pd.read_excel("data.xlsx")

    # 1.1
    # 由于大部分记录tel字段都为空，因此设置为默认值
    # 这里选取的默认值为市场监管总局政府信息公开办公室
    # 该单位联系电话 010—88650137

    data['tel'] = data['tel'].fillna("010—88650137")


    # 1.2
    # 处理 部门-职务
    # 首先剔除空字段
    # 定义分隔函数
    # 如果字段正常填写 那么返回一个二元组 分别对应部门与职务
    # 如果非法填写 那么返回第一个字段两次 用于填充
    # 加入部门、职务两个新字段 去除原字段

    data = data[data['部门-职务'].notnull()]

    def split2(i):
        res = str(i).split(" ")
        return (res[0], res[0]) if len(res) == 1 else (res[0], res[1])

    data['部门'] = data['部门-职务'].map(lambda i: split2(i)[0])
    data['职务'] = data['部门-职务'].map(lambda i: split2(i)[1])
    data = data.drop(['部门-职务'], axis=1)

    # 注册资金
    # 首先建立金钱单位与整数的映射关系
    # 利用正则表达式提取浮点数
    data = data[data['注册资金'].notnull()]
    def money(i):
        money_map = {"元": 1,
                     "百": 100,
                     "万": 10000,
                     "亿": 100000000}
        return int(float(re.findall(r"[-+]?\d*\.\d+|\d+", i)[0]) * money_map[i[-1]])

    data['注册资金'] = data['注册资金'].map(lambda i: money(i))

    # 公司年龄
    # 提取字段中的整数
    # 如果没有提取成功 例如'不足一年' 那么则返回 0
    data = data[data['公司年龄'].notnull()]

    def year(i):
        y = re.findall("\d+", i)
        return y[0] if len(y) > 0 else 0

    data['公司年龄'] = data['公司年龄'].map(lambda i: int(year(i)))

    # 读取当前日期格式 如2021/11/11
    # 处理日期格式问题 将其转化为2020-10-11的格式
    # 去除掉不符合格式的记录
    data['公司成立时间'] = pd.to_datetime(data['公司成立时间'], format='%Y/%m/%d', errors='coerce')
    data = data[data['公司成立时间'].notnull()]
    data['公司成立时间'] = data['公司成立时间'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # 1.3 数据概化
    # 职务等级
    def job_level(i):
        if '董事' in i or '主席' in i:
            return 'A'
        if '总经理'in i or '总裁' in i or '副总经理' in i:
            return 'B'
        if '总监' in  i or '副总监' in  i or'经理' in  i or '副经理' in  i or'主任' in  i or '主管' in i:
            return 'C'
        if '工程师' in  i or '员' in  i or '实习生' in i :
            return 'D'
        return 'E'

    data['职务等级'] = data['职务'].map(lambda i: job_level(i))

    # 工作类别
    def job_type(i):
        if '销售' in i or '市场' in i or '市场' in i or '客户' in i:
            return '市场类'
        if '业务' in i or '技术' in i or '项目' in i:
            return '技术类'
        if '营销' in i or '宣传' in i:
            return '营销类'
        if '财务' in i or '运营' in i or '人力' in i or '行政' in i:
            return '管理类'
        return '其他类'

    data['工作类别'] = data['职务'].map(lambda i: job_type(i))

    # 所在地区
    data['所在地区'] = cpca.transform(data['公司名称'])['省']

    # 公司类别
    def company_type(i):
        if '科技' in i or '软件' in i or '信息技术' in i:
            return '科技类'
        if '文化' in i or '传媒' in i or '广告' in i:
            return '文化传媒广告类'
        if '咨询' in i:
            return '咨询类'
        if '管理' in i:
            return '管理类'
        if '贸易' in i or '商贸' in i or '科贸' in i or '工贸' in i:
            return '贸易类'
        if '机械' in i or '设备' in i or '建筑' in i:
            return '制造业类'
        return '其他类'

    data['公司类别'] = data['公司名称'].map(lambda i: company_type(i))

    def money_level(i):
        if i < 10000000:
            return '1000万以下'
        elif i < 50000000:
            return '1000万以上5000万以下'
        elif i < 100000000:
            return '5000万以上1亿以下'
        else:
            return '1亿以上'

    data['注册资金等级'] = data['注册资金'].map(lambda i : money_level(i))

    data = data.reset_index()
    data = data.drop('index', axis=1)

    data.to_excel("final_data.xlsx", index=False)

    return data


def plt_chart(col, title):
    res = col.value_counts()
    plt.barh(res.keys().to_list(), res.values.tolist())
    plt.title(title+'-直方图')
    plt.tick_params(axis='y', labelsize=8)  # 设置x轴标签大小
    plt.tick_params(axis='x', labelsize=8)  # 设置x轴标签大小
    # plt.xticks(rotation=-90)
    global index
    plt.savefig(str(index)+title + '-直方图.jpg')
    index += 1
    plt.show()

    plt.pie(res.values.tolist(), labels=res.keys().to_list(), autopct='%3.2f%%')
    plt.title(title + '-饼图')
    plt.savefig(str(index)+title + '-饼图.jpg')
    index += 1
    plt.show()


def data_visualization(data):
    # 2.1 按照数据类型进行图表展示
    plt_chart(data['职务等级'], '职务等级')
    plt_chart(data['工作类别'], '工作类别')
    plt_chart(data['注册资金等级'], '注册资金等级')
    plt_chart(data['公司类别'], '公司类别')
    plt_chart(data['所在地区'],  '所在地区')


def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]


def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)


def correlation(x, y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x,y) / stddevx / stddevy


def plt_cov(col1, col2, col1_name, col2_name, title, scatter=True):
    if scatter:
        plt.scatter(col1, col2)
    else:
        y = [0, 0, 0, 0]
        for i,j in zip(col1, col2):
            if i==0 and j==0:
                y[0] += 1
            elif i==0 and j==1:
                y[1] += 1
            elif i==1 and j==0:
                y[2] += 1
            else:
                y[3] += 1
        x = ["", "", "", ""]
        x[0] = ('%s %s -- %s %s') % ('1', '否', '2', '否')
        x[1] = ('%s %s -- %s %s') % ('1', '否', '2', '是')
        x[2] = ('%s %s -- %s %s') % ('1', '是', '2', '否')
        x[3] = ('%s %s -- %s %s') % ('1', '是', '2', '是')

        plt.bar(x, y)
    plt.xlabel(col1_name)
    plt.ylabel(col2_name)
    plt.title(title)
    global index
    plt.savefig(str(index) + title + '.jpg')
    index += 1
    plt.show()
    # 计算皮尔逊矩阵系数
    r = correlation(col1, col2)
    print('%s - %s (%s相关) : Pearson积矩系数 - %f' % (col1_name, col2_name, ('正' if r > 0 else '负'), r))


def cov_analyze(data):
    # 公司注册资金与公司年龄的关系
    temp = data.sort_values(by=['注册资金'])[500:6000]
    plt_cov(temp['注册资金'], temp['公司年龄'], '注册资金', '公司年龄', '注册资金-公司年龄')

    # 用户是否认证与公司是否认证之间的关联关系
    plt_cov(data['是否认证'], data['公司是否认证'], '是否认证', '公司是否认证', '是否认证-公司是否认证', False)



def plt_diff(col, col2, title):
    res = col.value_counts()
    plt.barh(res.keys().to_list(), res.values.tolist(), alpha=0.6)

    res = col2.value_counts()
    plt.barh(res.keys().to_list(), res.values.tolist())

    plt.title(title + '-直方图')

    global index
    plt.savefig(str(index) + title + '-直方图.jpg')
    index += 1
    plt.show()

def diff_analyze(data):
    # 认证用户与非认证用户 职务分布情况 差异
    df1 = data[data['是否认证'] == 1]['职务等级']
    df2 = data[data['是否认证'] == 0]['职务等级']

    plt_diff(df1, df2, '认证用户与非认证用户 职务分布情况 差异')


    # 认证公司和非认证公司 注册资金 差异
    df1 = data[data['公司是否认证'] == 1]['注册资金等级']
    df2 = data[data['公司是否认证'] == 0]['注册资金等级']

    plt_diff(df1, df2, '认证公司和非认证公司 注册资金 差异')


    # 认证公司和非认证公司 公司年龄 差异
    df1 = data[data['公司是否认证'] == 1]['公司年龄']
    df2 = data[data['公司是否认证'] == 0]['公司年龄']

    plt_diff(df1, df2, '认证公司和非认证公司 公司年龄 差异')


if __name__ == '__main__':
    data = clean()

    data_visualization(data)
    #
    cov_analyze(data)
    #
    diff_analyze(data)