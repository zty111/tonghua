from keras.backend import learning_phase


ker_num = 6 # 同时运行的进程数

num = 1 # 每个进程生成棋局数

mcts_num = 160 # 搜索数

thread_num = 8 # 同时运行线程数

learning_rate = 0.0001

show = False #训练中是否显示详细对局

bot_name = 'BotZero3.h5' #AI名称

#agent中搜索st可以看一次预测时间和一步计算时间
