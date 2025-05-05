import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(file, pattern):
    # 初始化空列表来存储提取的数据
    data = []

    # 正则表达式来匹配时间和schedule、comm、swap、execute等信息
    pattern = re.compile(pattern)

    # 读取日志文件
    with open(file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp = match.group(1)
                schedule_time = float(match.group(2))
                comm_time = float(match.group(3))
                swap_time = float(match.group(4))
                execute_time = float(match.group(5))
                # 将提取的数据添加到列表中
                data.append([timestamp, schedule_time, comm_time, swap_time, execute_time])

    return data


def plot1():
    # 将数据转换为DataFrame
    df = pd.DataFrame(get_data(
        file='log1.txt',
        pattern=r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*avg ([\d.]+)%.*avg ([\d.]+)%.*avg ([\d.]+)%.*avg ([\d.]+)%"
    ), columns=['Timestamp', 'Schedule', 'Comm', 'Swap', 'Execute'])

    # 将时间戳转换为pandas datetime格式
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m-%d %H:%M:%S')

    # 每10秒钟计算平均值
    df.set_index('Timestamp', inplace=True)
    df_resampled = df.resample('10S').mean()

    # 绘制图表
    plt.figure(figsize=(7, 4))
    plt.plot(df_resampled.index, df_resampled['Schedule'], label='Schedule')
    plt.plot(df_resampled.index, df_resampled['Comm'], label='Comm')
    plt.plot(df_resampled.index, df_resampled['Swap'], label='Swap')
    plt.plot(df_resampled.index, df_resampled['Execute'], label='Execute')

    plt.xlabel('Timestamp')
    plt.ylabel('Time (%)')
    plt.title('Schedule, Comm, Swap, Execute Times Over 10s Intervals')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示图表
    plt.show()


def plot2():
    # 将数据转换为DataFrame
    df = pd.DataFrame(get_data(
        file='log2.txt',
        pattern=r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\(([\d.]+)%\).*\(([\d.]+)%\).*\(([\d.]+)%\).*\(([\d.]+)%\)"
    ), columns=['Timestamp', 'Schedule', 'Comm', 'Swap', 'Execute'])

    print(df)

    plt.figure(figsize=(7, 4))

    # Define the columns to plot
    columns_to_plot = ['Schedule', 'Comm', 'Swap', 'Execute']

    # Plot the CDF for each column
    for column in columns_to_plot:
        # Sort the values in the column
        sorted_values = np.sort(df[column])
        # Calculate the CDF
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        # Plot the CDF
        plt.plot(sorted_values, cdf, label=column)

    # Add labels and title
    plt.xlabel('Value (%)')
    plt.ylabel('CDF')
    plt.title('CDF of Schedule, Comm, Swap, and Execute over 1381 iterations')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot3():
    # Data for task_rate=50, num_tasks=100
    labels_1 = ['truncated distribution', 'ground truth', 'only mean', 'fcfs']
    avg_1 = [51.69926567554474, 50.50222090244293, 50.92292751789093, 60.29078952074051]

    # Data for task_rate=3, num_tasks=180
    labels_2 = ['truncated distribution', 'ground truth', 'only mean', 'fcfs']
    avg_2 = [51.6507867746883, 48.28033616940181, 51.559814990891354, 59.91505684985055]

    # Plot for task_rate=50, num_tasks=100
    plt.figure(figsize=(5, 3))
    plt.bar(labels_1, avg_1, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Method')
    plt.ylabel('Average')
    plt.title('Average for task_rate=50, num_tasks=100')
    plt.ylim(0, 65)  # Set y-axis limits
    plt.show()

    # Plot for task_rate=3, num_tasks=180
    plt.figure(figsize=(5, 3))
    plt.bar(labels_2, avg_2, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Method')
    plt.ylabel('Average')
    plt.title('Average for task_rate=3, num_tasks=180')
    plt.ylim(0, 65)  # Set y-axis limits
    plt.show()


if __name__ == '__main__':
    plot1()
    plot2()
    # plot3()

