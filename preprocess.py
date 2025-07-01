import re
import shutil
import os

def select_7k_images(c_type='humor'):
    '''30k -> 7k'''
    # open data/type/train.p
    with open('data/' + c_type + '/train.p', 'r') as f:
        res = f.readlines()

    # extract img names
    img_list = []
    r = re.compile("\d*_")
    for line in res:
        if len(line) < 10:
            continue
        line = r.search(line)
        line = line.group(0)[:-1] + '.jpg'
        img_list.append(line)

    # copy imgs
    for img_name in img_list:
        shutil.copyfile('data/flickr30k_images/' + img_name,
                        'data/flicker_8k/Flicker8k_Dataset/' + img_name)

def select_factual_captions():
    '''30k -> 7k'''
    # get filenames in flickr7k_images
    filenames = os.listdir('data/flicker_8k/Flicker8k_Dataset')
    # open data/results_20130124.token
    with open('data/factual_train.txt', 'r') as f:
        res = f.readlines()

    # write out
    with open('data/factual_train.txt', 'w') as f:
        r = re.compile('\d*.jpg')
        for line in res:
            img = r.search(line)
            img = img.group(0)
            if img in filenames:
                f.write(line)

def select_romantic_captions():
    '''30k -> 7k'''
    # get filenames in flickr7k_images
    filenames = os.listdir('data/flicker_8k/Flicker8k_Dataset')
    # open data/results_20130124.token
    with open('data/romantic_train.txt', 'r') as f:
        res = f.readlines()

    # write out
    with open('data/romantic_train.txt', 'w') as f:
        r = re.compile('\d*.jpg')
        for line in res:
            img = r.search(line)
            img = img.group(0)
            if img in filenames:
                f.write(line)

def select_humorous_captions():
    '''30k -> 7k'''
    # get filenames in flickr7k_images
    filenames = os.listdir('data/flicker_8k/Flicker8k_Dataset')
    # open data/results_20130124.token
    with open('data/humorous_train.txt', 'r') as f:
        res = f.readlines()

    # write out
    with open('data/humorous_train.txt', 'w') as f:
        r = re.compile('\d*.jpg')
        for line in res:
            img = r.search(line)
            img = img.group(0)
            if img in filenames:
                f.write(line)


# if __name__ == '__main__':
    # select_7k_images()
    # select_factual_captions()
    # select_romantic_captions()
    # select_humorous_captions()
