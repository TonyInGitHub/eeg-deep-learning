import tensorflow as tf

NTIMEPOINTS= 2560
NCHANNELS = 3
NCLASSES = 2
LABEL_BYTES = 3

### INPUT_EEG
# defines pipelines for EEG data import and queuing

### INPUT
def decode_file(line):
    record_bytes = tf.decode_raw(line, tf.float32)
    record_bytes = tf.reshape(record_bytes, [NCHANNELS*NTIMEPOINTS+LABEL_BYTES])
    # from the bytes that were read isolate label and image bytes
    image = tf.strided_slice(record_bytes, [0], [NCHANNELS*NTIMEPOINTS], name = "slice_data")
    label = tf.strided_slice(record_bytes, [NCHANNELS*NTIMEPOINTS], [NCHANNELS*NTIMEPOINTS+1], name = "slice_label")
    animal_id =  tf.strided_slice(record_bytes, [NCHANNELS*NTIMEPOINTS+1], [NCHANNELS*NTIMEPOINTS+2], name = "slice_animal_id")
    time_sz =  tf.strided_slice(record_bytes, [NCHANNELS*NTIMEPOINTS+2], [NCHANNELS*NTIMEPOINTS+3], name = "time_to_sz")
    # a bit of processing on image and label
    image = tf.reshape(image, [NCHANNELS*NTIMEPOINTS])
    # image = tf.reshape(image, [NCHANNELS, NTIMEPOINTS])
    # # # zscoring of "image"
    # _, var = tf.nn.moments(image, axes=[1], keep_dims = True)
    # image = tf.reshape(image/tf.sqrt(var), [NCHANNELS*NTIMEPOINTS])
    #image = tf.reshape(image, [NCHANNELS*NTIMEPOINTS])
    image = tf.cast(image, tf.float32)
    # normalize image ?!!!
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label,NCLASSES);
    label = tf.reshape(label, [NCLASSES])
    animal_id = tf.cast(animal_id, tf.float32)
    time_sz = tf.cast(time_sz, tf.float32)
    return image, label, animal_id, time_sz

def input_pipeline(filenames, batch_size, bShuffle):
    dataset_train = tf.contrib.data.FixedLengthRecordDataset(filenames, record_bytes=4*(NCHANNELS*NTIMEPOINTS+LABEL_BYTES))
    dataset_train = dataset_train.map(decode_file)
    if bShuffle:
        dataset_train = dataset_train.shuffle(buffer_size=5000)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.batch(batch_size)
    iterator = dataset_train.make_one_shot_iterator()
    image_batch, label_batch, animal_id_batch, time_sz_batch = iterator.get_next()

    return image_batch, label_batch, animal_id_batch, time_sz_batch
