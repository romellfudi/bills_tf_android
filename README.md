```sh
# sugguest mobilenet model
version_string { '1.0', '0.75', '0.50', '0.25' }
size_string { '224', '192', '160', '128' }
# suggest inception model
input_layer = "input"
output_layer = "InceptionV3/Predictions/Reshape_1"
to:
input_layer = "Mul"
output_layer = "final_result"
```

# Mobilenet
## Retrain
```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
IMAGE_DIR="bills_photos"
python scripts/retrain.py \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_mobilenet_graph.pb \
  --output_labels=tf_files/retrained_mobilenet_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir="${IMAGE_DIR}"
```
## Labeling
```
python scripts/label_image.py \
  --graph=tf_files/retrained_mobilenet_graph.pb  \
  --labels=tf_files/retrained_mobilenet_labels.txt \
  --image=bills_photos/Billetes\ 50/P9030055.JPG
python scripts/label_image.py \
    --graph=tf_files/retrained_mobilenet_graph.pb  \
  --labels=labels.txt \
    --image=bills_photos/Billetes\ 10/P9030043.JPG
```
## TFLite
```
IMAGE_SIZE=224
tflite_convert \
  --graph_def_file=tf_files/retrained_mobilenet_graph.pb \
  --output_file=tf_files/optimized_mobilenet_graph.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT
```

# INCEPTION v3
## Retrain
```
ARCHITECTURE="inception_v3"
IMAGE_SIZE=299
IMAGE_DIR="bills_photos"
python scripts/retrain.py \
  --bottleneck_dir=tf_files/inception_bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/inception_retrained_graph.pb \
  --output_labels=tf_files/inception_retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir="${IMAGE_DIR}"
```
## Labeling
```sh
    # --input_mean=0 \
    # --input_std=255 \
python scripts/label_image.py \
    --graph=tf_files/inception_retrained_graph.pb  \
  --labels=tf_files/inception_retrained_labels.txt \
    --input_height=299 \
    --input_width=299 \
    --input_layer="Mul" \
    --image=bills_photos/Billetes\ 50/P9030055.JPG

python scripts/label_image.py \
    --graph=tf_files/inception_retrained_graph.pb  \
  --labels=tf_files/inception_retrained_labels.txt \
    --input_height=299 \
    --input_width=299 \
    --input_mean=0 \
    --input_std=255 \
    --input_layer="Mul" \
    --image=bills_photos/Billetes\ 10/P9030043.JPG
```

```
IMAGE_SIZE=299
tflite_convert \
  --graph_def_file=tf_files/inception_retrained_graph.pb \
  --output_file=tf_files/inception_optimized_graph.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
  --input_array=Mul \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT
```