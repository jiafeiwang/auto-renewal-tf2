# auto-renewal-tf2
<p align="left">
  <img src='https://img.shields.io/badge/python-3.7-blue'>
  <img src='https://img.shields.io/badge/tensorflow-2.1.0-brightgreen'>
  <img src='https://img.shields.io/badge/keras-2.2.4-brightgreen'>
</p>  

自动续费，如月缴会员场景：   
开通时签约微信或支付保自动续费，每个月到扣费日（D）进行自动扣费，连续扣30日，直至扣费成功，D+30后扣费不成功会员订单失效；  
扣费后用户在[D, D+30]内可以选择关闭自动续费或申请退费，关闭自动续费或退费后会员订单失效。  

本项目搭建了DNN模型用来预估订单D~D+30日内（output_dim=31）的扣费成功（P）、关闭自动续费（C）、退费（R）的概率分布，模型结构类似于PNN，  
product方式为Inner product，deep部分为MLP；  
在进入MLP前对用户行为序列进行注意力汇聚，与product、embedding、dense特征concat后一起进入MLP。  

对于每一期的P、C、R需要分别筛选数据各训练一个模型，预测出订单30天的概率分布后，根据订单所处的当前期的天数（days），聚合得到实际的rate;  
聚合方式为：
D,   D+1, D+2,  D+3, ... D+29,    N  
0.3, 0.1, 0.02, 0.01, ...0.0001,  0.5  

days = 1  
rate = (0.02 + 0.01 + ... + 0.0001)/(0.02 + 0.01 + ... 0.0001 + 0.5)  

days = 2  
rate = (0.01 + ... + 0.0001)/(0.01 + ... 0.0001 + 0.5)  

然后通过rate和订单价格计算订单维度总价值，按特征维度聚合进行使用。
