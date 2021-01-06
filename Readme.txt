用于x20鳞状细胞数量检测
name:18.0 18_0.pb
time: 0.0523 
Precious: 88.81     
Recall: 87.24    
F1_Measure: 88.02
training   data :387
validation data :67
model:18_1.pb  size: 11.3M
下限：671 M   |   上限：4941 M
输入tensor_name:
    'input_0:0’    	shape=[1,None,None,3], dtype=tf.uint8
    'get_info:0'	shape=None, dtype=tf.int32
输出tensor_name:
    'output:0'	shape=[1,None,3] 	                #每一行分别表示：点的横坐标x，纵坐标y，类别。 其中类别index 1分别阳性。
    'info:0'	shape=[None],dtype=tf.string		#模型的版本信息
