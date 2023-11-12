from PIL import Image
import numpy as np
import tensorflow as tf

##########모델 로드

interpreter = tf.lite.Interpreter(model_path='/Users/yongcho/dev/yonggit/eyeway_ai/ultralytics/ultralytics/yong/best_float32.tflite')


##########모델 예측

image = Image.open('/Users/yongcho/dev/yonggit/eyeway_ai/ultralytics/ultralytics/yong/subway_data/v1/test/images/057739B1-38B8-4C72-B365-630953DE23AC_1_105_c_jpeg.rf.0dc606445ec67ab26a1b32d8c92c2b10.jpg')
image = image.resize((640, 640))
numpy_image = np.array(image) #이미지 타입을 넘파이 타입으로 변환
x_test = np.array([numpy_image])
x_test = x_test.astype(dtype=np.float32)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], x_test)
interpreter.invoke()

output_details = interpreter.get_output_details()
y_predict = interpreter.get_tensor(output_details[0]['index'])
#y_predict = (y_predict - 0) * 0.00390625
y_predict = (y_predict - output_details[0]['quantization'][1]) * output_details[0]['quantization'][0]
#y_predict = (y_predict - output_details[0]['quantization_parameters']['zero_points'][0]) * output_details[0]['quantization_parameters']['scales'][0]

confidence = y_predict[0][y_predict[0].argmax()]
print(confidence) #sports car 0.7890625