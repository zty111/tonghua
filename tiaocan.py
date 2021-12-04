from keras.backend import learning_phase


ker_num = 5 # 同时运行的进程数

num = 4 # 每个进程生成棋局数
rand_num = 1000 # 随机限制数
mcts_num = 160 # 搜索数

thread_num = 8 # 同时运行线程数

learning_rate = 0.01

show = False #训练中是否显示详细对局

bot_name = 'Bot.h5' #AI名称