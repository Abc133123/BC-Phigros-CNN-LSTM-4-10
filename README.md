# AI-Play-Phigros-BehavioralCloning-CNN-LSTM

模仿学习动作克隆模型 打了4/10麻了 音游AI 使用win电脑模拟器截图winapi控制鼠标点击 训练集纯冰
闲的 耗时3-5小时  踩数据收集的坑了 再看看其他方法。<br>
原训练集就这样是坑（错误方法）上的 推荐自行判断<br>
先模型无实际功能 效果只能拿10/3分 和猫瞎抓差不多<br>
但是这是我懒得修的情况下 改进空间很大 <br>
该项目现价值是跑了手机ADB到电脑模拟器上截图控制的坑 和基本CNN训练流程<br>

<h2>快速开始</h2>
<h5>环境依赖<br>
Python 3.10+<br>
PyTorch, OpenCV, mss, pyautogui, numpy<br>
1. 数据采集
运行采集脚本，连接手机 ADB，在模拟器中操作。<br>
模拟器推荐mumu模拟器<br>
工作窗口附着win左中占一半 自己看脚本设置<br>
ADB和设备名称路径需要改<br>
  设备名称:参考指令 adb.exe devices<br>
python sj3.py<br>
数据保存在 phigros_data 目录。<br>

2. 模型训练 (可选)
基于采集数据训练模型。<br>

python main.py<br>
模型保存在 models/best.pth。<br>

3. 运行推理 python godj.py<br>
使用训练好的模型预测操作。<br>
数据采集：通过 ADB 同步手机触摸事件到 PC，记录操作序列与屏幕画面。<br>
视觉方案：基于颜色空间过滤检测音符，计算落点时间并触发点击。<br>
深度学习：CNN 提取画面特征，LSTM 学习时序规律，预测下一帧操作。<br>

