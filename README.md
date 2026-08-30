# Face Recognition Project

本地实时多人脸识别项目，使用 InsightFace 完成人脸检测与身份匹配。

安全功能包括：

- 每张实体脸独立跟踪
- 眨眼与转头主动活体验证
- MiniFASNet V1SE + V2 双模型静默防伪
- 打印照片与屏幕翻拍拦截
- Core ML GPU / Neural Engine 加速
- 多人日志加锁批量写入

## 隐私

人脸特征文件、识别日志和本地虚拟环境不会提交到 Git。摄像头画面和人脸特征只在本机处理，程序没有网络上传代码。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python recognize.py
```

首次使用前，通过 `enroll_person.py` 在本地生成人脸特征文件。

## 防伪模型

项目包含 Apache 2.0 授权的 MiniFASNet ONNX 模型。详细来源和校验值见 `THIRD_PARTY_NOTICES.md`。
