
import tensorflow as tf
import numpy as np
from create_verification_code import gen_captcha_text_and_image

from train import crack_captcha_cnn, vec2text, convert2gray, keep_prob
from train import MAX_CAPTCHA, CHAR_SET_LEN, X


def crack_captcha():
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('models'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        count_out = 0
        right_number = 0
        while(True):
            root_text, image = gen_captcha_text_and_image()
            image = convert2gray(image)
            captcha_image = image.flatten() / 255
            text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * CHAR_SET_LEN + n] = 1
                i += 1
            predict_text = vec2text(vector)
            print("正确: {}  预测: {}".format(root_text, predict_text))
            if count_out == 1000:
                break
            count_out += 1

crack_captcha()
