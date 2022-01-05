import argparse
import os
import torch

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, required=False, default='dataset', help='data them')
parser.add_argument('--root_path', type=str, default='./data/CPU/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='dataset.csv', help='data file path')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='cpu_util_percent', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


parser.add_argument('--seq_len', type=int, default=168, help='编码器的输入序列长度（默认为 96）input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=96, help='解码器的起始令牌长度（默认为 48）start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='预测序列长度（默认为 24）prediction sequence length')

parser.add_argument('--enc_in', type=int, default=1, help='编码器输入大小（默认为 7）encoder input size') # 维度
parser.add_argument('--dec_in', type=int, default=1, help='解码器输入大小（默认为 7）decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='输出大小（默认为 7）output size')

parser.add_argument('--d_model', type=int, default=512, help='模型尺寸（默认为 512）dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='头数（默认为 8）num of heads') # 8 或者 16
parser.add_argument('--e_layers', type=int, default=2, help='编码器层数（默认为 2）num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='解码器层数（默认为 1）num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='堆栈编码器层数（默认为3,2,1)num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='fcn 的维度（默认为 2048）dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='概率稀疏 attn 因子（默认值为 5）probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='填充类型（默认为 0）padding type')
parser.add_argument('--distil', action='store_false', help='是否在编码器中使用蒸馏，使用此参数意味着不使用蒸馏（默认为True)whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='（默认0.05）dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='激活函数（默认为gelu)activation')
parser.add_argument('--output_attention', action='store_true', help='是否在编码器中输出注意力，使用此参数意味着输出注意力（默认为False)whether to output attention in ecoder')
"""
enc_in: informer的encoder的输入维度
dec_in: informer的decoder的输入维度
c_out: informer的decoder的输出维度
d_model: informer中self-attention的输入和输出向量维度
n_heads: multi-head self-attention的head数
e_layers: informer的encoder的层数
d_layers: informer的decoder的层数
d_ff: self-attention后面的FFN的中间向量表征维度
factor: probsparse attention中设置的因子系数
padding: decoder的输入中，作为占位的x_token是填0还是填1
distil: informer的encoder是否使用注意力蒸馏
attn: informer的encoder和decoder中使用的自注意力机制
embed: 输入数据的时序编码方式
activation: informer的encoder和decoder中的大部分激活函数
output_attention: 是否选择让informer的encoder输出attention以便进行分析

小数据集的预测可以先使用默认参数或适当减小d_model和d_ff的大小

"""

parser.add_argument('--do_predict', action='store_true', help='是否预测看不见的未来数据，使用此参数意味着进行预测（默认为False)whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='实验次数（默认为 2）experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='训练纪元（默认为 6）train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批大小（默认为 32）batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
# parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
# 想要获得最终预测的话这里应该设置为True；否则将是获得一个标准化的预测。
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

# 进行parser的变量初始化，获取实例。
args = parser.parse_args()

# 判断GPU是否能够使用，并获取标识
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 判断是否使用多块GPU，默认不使用多块GPU
if args.use_gpu and args.use_multi_gpu:
    # 获取显卡列表，type：str
    args.devices = args.devices.replace(' ','')
    # 拆分显卡获取列表，type：list
    device_ids = args.devices.split(',')
    # 转换显卡id的数据类型
    args.device_ids = [int(id_) for id_ in device_ids]
    # 获取第一块显卡
    args.gpu = args.device_ids[0]

# 初始化数据解析器，用于定义训练模式、预测模式、数据粒度的初始化选项。
"""
字典格式：{数据主题：{data：数据路径,'T':目标字段列名,'M'：，'S'：，'MS':}}

'M:多变量预测多元（multivariate predict multivariate）'，
'S:单变量预测单变量（univariate predict univariate）'，
'MS:多变量预测单变量（multivariate predict univariate）'。
"""
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'dataset':{'data':"dataset.csv",'T':'cpu_util_percent','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}
# print(args.data)
# print(args.data_path)
# 判断在parser中定义的数据主题是否在解析器中
if args.data in data_parser.keys():
    # 根据args里面定义的数据主题，获取对应的初始化数据解析器info信息，type：dict
    data_info = data_parser[args.data]
    # 获取该数据主题的数据文件的路径
    args.data_path = data_info['data']
    # 从数据解析器中获取 S或MS任务中的目标特征列名。
    args.target = data_info['T']
    # 从数据解析器中 根据变量features的初始化信息 获取 编码器输入大小，解码器输入大小，输出尺寸
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]


# 堆栈编码器层数，type：list
args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
# 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):

    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        # 预测看不见的数据
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
