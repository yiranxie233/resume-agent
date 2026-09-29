from paddleocr import PaddleOCR
import os

os.environ['PADDLE_PDX_HOME'] = "./weights"
os.makedirs('./weights', exist_ok=True)

# 默认使用 PP-OCRv6 模型
ocr = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型

    # 关键：指定检测、识别模型目录，不存在会自动下载到这里
    # text_detection_model_dir="./weights/ppocrv6_medium_det",
    # text_recognition_model_dir="./weights/ppocrv6_medium_rec",
)
# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv5") # 通过 ocr_version 参数切换为 PP-OCRv5 版本
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数切换为 PP-OCRv4 版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # 使用 PP-OCRv5 的 server 模型
result = ocr.predict("photos/photo2.jpg")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")