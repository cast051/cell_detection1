import tensorflow as tf
from config import get_config
import os
from dataloader import dataloader
from evalute import validation_model
import numpy as np
import cv2
import glob
import imageio

os.environ['CUDA_VISIBLE_DEVICES']='4'



def main():
    #get config
    config=get_config(is_training=False)

    # load post process op
    postprocess_module = tf.load_op_library(config.so_path)

    # create session and load model
    ConfigProto = tf.ConfigProto()
    ConfigProto.gpu_options.per_process_gpu_memory_fraction = 0.03 # 占用GPU40%的显存
    # ConfigProto.gpu_options.allow_growth = True  #最小的GPU显存使用量，动态申请显存
    sess = tf.Session(config=ConfigProto)
    sess.run(tf.global_variables_initializer())

    #load model
    with tf.gfile.FastGFile(config.pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input, output, net_output, info, get_info = tf.import_graph_def(graph_def,
                      return_elements=["input:0", "output:0","net_output:0", "info:0","get_info:0"])

    info_ = sess.run([info], feed_dict={get_info: 0})
    print(info_)

    if config.testmodel=="test1":
        # load dataset
        _, valid_records = dataloader.get_datasetlist(config.data_dir)
        validation_dataset_reader = dataloader(valid_records, augument_flag=False)
        validation_model(sess, valid_records, validation_dataset_reader, net_output, output, input,config.log_test_dir)

    elif config.testmodel=="test2":
        img_list = glob(config.test2_img_path + '/*.jpg')
        for k, img_path in enumerate(img_list):
            print(img_path)
            path, name = os.path.split(img_path)
            img_name = os.path.splitext(name)[0]
            img = imageio.imread(img_path)
            img = np.expand_dims(img, 0)
            output_post, net_output_ = sess.run([output, net_output], feed_dict={input: img})
            output_post = np.squeeze(output_post, 0)

            # debug
            print(len(output_post))
            image_co = img[0].astype(np.uint8)
            for j in range(output_post.shape[0]):
                if output_post[j][2] == 1:
                    pointx1 = output_post[j][0].astype(np.uint)
                    pointy1 = output_post[j][1].astype(np.uint)
                    cv2.circle(image_co, (pointx1, pointy1), 3, (0, 255, 0), -1)
            imageio.imwrite(path + "/" + img_name + '_co.png', image_co)
            imageio.imwrite(path + "/" + img_name + '_netout.png', (255 * net_output_[0]).astype(np.uint8))
            print("Saved image: %d" % k)




if __name__ == '__main__':
    main()
