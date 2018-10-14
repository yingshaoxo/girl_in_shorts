# girl_in_shorts
This is a detector for detecting if a girl wearing shorts.

### It's highly rely on tensorflow(https://github.com/tensorflow/tensorflow)
```
git clone https://github.com/tensorflow/tensorflow.git
git clone https://github.com/yingshaoxo/girl_in_shorts.git
```

### Retain image classifer
```
python3 tensorflow/tensorflow/examples/image_retraining/retrain.py --model_dir=tensorflow/tf_files/inception-v3 --output_graph=tensorflow/tf_files/girl_classify.pb --output_labels=tensorflow/tf_files/girl_classify_labels.txt --image_dir=girl_in_shorts/data --bottleneck_dir=tensorflow/tf_files/girl_bottleneck/
```

### After you got `tensorflow/tf_files/girl_classify.pb`
```
python3 label_from_screen.py
```
