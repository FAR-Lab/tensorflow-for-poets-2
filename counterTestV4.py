#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 07:16:04 2018
@author: raghav prabhu
Re-modified TensorFlow classification file according to our need.
"""
import tensorflow as tf
import sys
import os
import csv
import time
import re

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def _parse_function(filename,label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    float_caster = tf.cast(image_decoded, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [224, 224])
    normalized = tf.divide(tf.subtract(resized, [128]), [128])
    return normalized, label
    #
    # dims_expander = tf.expand_dims(float_caster, 0);
    # resized = tf.image.resize_bilinear(dims_expander, [224, 224])


    # float_caster = tf.cast(blub, tf.float32)
    # dims_expander = tf.expand_dims(float_caster, 0);
    # resized = tf.image.resize_bilinear(dims_expander, [224, 224])
    # normalized = tf.divide(tf.subtract(resized, [128]), [128])
    #  normalized

'''
Classify images from test folder and predict dog breeds along with score.
'''
def classify_image(image_path, headers,output_filename):
    f = open(output_filename+".csv",'w')
    f2 = open(output_filename+"_clips_.csv",'w')
    writer = csv.DictWriter(f, fieldnames = headers)
    writer.writeheader()
    writer2 = csv.DictWriter(f2, fieldnames = ['clipNumber','p'])
    writer2.writeheader()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    files = os.listdir(image_path)
    complete_files = [image_path+'/'+file for file in files]
    ds = tf.data.Dataset.from_tensor_slices((complete_files, files))
    ds = ds.map(_parse_function)
    iterator = ds.make_initializable_iterator()
    next_element = iterator.get_next()
    clipClassificationList=[]
    lastClip=-1
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        i=0
        while True:
            i+=1
            try:
                start = time.time()


                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                input_tensor = sess.graph.get_tensor_by_name('input:0')

                temp = sess.run(next_element)
                predictions = sess.run(softmax_tensor, {input_tensor: temp[0]})
                end=time.time()
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                records = []
                row_dict = {}
                #head, tail = os.path.split(file)\\
                clipNumber= int(temp[1].split("_")[0])
                row_dict['clip'] = clipNumber

                if(lastClip!=clipNumber):
                    print('\n----\nEvaluation time (1-image): {0:.3f}s at number{1}'.format(end-start,i))
                    if(lastClip!=-1):
                        print("We just finished clip {0}, now going to {1}.\n----\n".format(lastClip,clipNumber))
                        output=[]
                        row_dictionary_temp={}
                        row_dictionary_temp['clipNumber']=lastClip
                        row_dictionary_temp['p'] = max(set(clipClassificationList),key=clipClassificationList.count)
                        output.append(row_dictionary_temp.copy())
                        writer2.writerows(output)
                        clipClassificationList=[]
                        lastClip=clipNumber
                    else:
                        lastClip=clipNumber
                frameNumber = temp[1].split("_")[1][:-4]
                frameNumber = re.sub('image', '', frameNumber)
                row_dict['frame'] = frameNumber
                row_dict['p']= label_lines[top_k[0]]
                clipClassificationList=label_lines[top_k[0]]
                for node_id in top_k:
                    human_string = label_lines[node_id]
                    # Some breed names are mismatching with breed name in csv header names.
                    human_string = human_string.replace(" ","_")
                    score = predictions[0][node_id]
                    #print('%s (score = %.5f)' % (human_string, score))
                    row_dict[human_string] = score
                if(not i % 100):
                    print('Evaluation time (1-image): {0:.3f}s at number{1}'.format(end-start,i))
                records.append(row_dict.copy())
                writer.writerows(records)
            except tf.errors.OutOfRangeError:
                print("We should be done now")
                output=[]
                row_dictionary_temp={}
                row_dictionary_temp['clipNumber']=lastClip
                row_dictionary_temp['p'] = max(set(clipClassificationList),key=clipClassificationList.count)
                output.append(row_dictionary_temp.copy())
                writer2.writerows(output)
                print("Just calculated the last clip statistics")
                f.flush()
                f2.flush()
                f.close()
                f2.close()
                break

def main():
    test_data_folder = sys.argv[1]

    #template_file = open('sample_submission.csv','r')
    #d_reader = csv.DictReader(template_file)

    #get fieldnames from DictReader object and store in list
    #headers = d_reader.fieldnames
    #template_file.close()

    #classify_image(test_data_folder, headers)
    headers =['clip','frame','n','o','y','p']
    classify_image(test_data_folder, headers, sys.argv[2] )

if __name__ == '__main__':
    main()
