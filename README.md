# Diffusion

## Requirements

- [loguru](https://github.com/Delgan/loguru), [yacs](https://github.com/rbgirshick/yacs), requests
- [PyTorch](https://pytorch.org/)
- diffusers, datasets

## 说明

目录结构如下（省略了每个目录中的__init__.py）

```shell
├──  config
│    └── defaults.py  - 默认配置文件
│    └── test_config.yml  - 实验配置文件
│ 
│
├──  data  
│    └── datasets  - 数据文件目录
│    └── transforms  - 预处理用的一切函数
│    └── build.py     - 产生data loader
│
│
├──  engine
│   ├── trainer.py     - 定义训练过程
│   └── inference.py   - 定义推理过程
│
│
│
├── modeling            - 该目录下定义模型
│   └── example_model.py
│
│
├── solver             
│   └── build.py           - 产生求解器
│   └── lr_scheduler.py    - 定义学习率调度器
│   
│ 
├── run             - 该目录下存放实际运行的脚本
│   └── train_net.py
│   
│ 
└── utils            - 定义实用工具
     ├── logger.py
     └── any_other_utils_you_need
```

## hugging face 数据集使用

在[hugging face](https://huggingface.co/settings/tokens)获取令牌，然后在终端通过huggingface-cli login来输入，之后才能使用数据和模型。

read权限允许你访问公开的数据集、模型和其他资源，下载这些资源并进行使用。write权限除了具备read权限的所有功能外，还允许你上传自己的数据集、模型或文件到Hugging Face Hub，创建新的仓库或修改已有的仓库。
