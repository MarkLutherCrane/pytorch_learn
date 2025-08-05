# pytorch_learn
小土堆 -- 学习用


* 本机protobuf版本过高，与tensorboard不兼容
```
pip uninstall -y tensorboard protobuf 
pip install tensorboard==2.14.0 protobuf==3.20.3
```

* tensorboard命令
```chatinput
tensorboard --logdir=logs --port=6007  # logs是指定的路径
```