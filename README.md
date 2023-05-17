## 手势识别

例子是x东的

原图片 和 标注图片 分别保存在imgs/target和imgs/label

启动main.py 运行脚本训练  运行失败 可以把 torch.permute(image, (2, 0, 1)) 替换为torch.permute(image, (0, 1,2))

test.py 查看结果 需要注意opencv版本
运行失败可以
torch.permute(transform, (2, 0, 1))
这里交换维度 改为
torch.permute(transform, (0, 1, 2))


使用EISeg工具去标注


https://blog.csdn.net/qq_37541097/article/details/120154543

 ![image](test.jpg)

欢迎加入星球哦
星球链接https://t.zsxq.com/06bIUvBEM
