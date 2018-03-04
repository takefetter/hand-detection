gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --runtime-version 1.4 \
    --job-dir=gs://takefetter_hand_detector/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --config object_detection/samples/cloud/cloud.yml \
    -- \
    --train_dir=gs://takefetter_hand_detector/train \
    --pipeline_config_path=gs://takefetter_hand_detector/data/ssd_mobilenet_v1_hand.config
