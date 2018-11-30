import cv2
import argparse
import os
import imageio
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import scipy.ndimage
import matplotlib.pyplot as plt
import maxflow

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import animation

def get_file_name(prefix, sigma_s_val, sigma_r_val, shade_factor_val = 0.0):
    v = str(int(10000*sigma_s_val + (100*sigma_r_val + shade_factor_val)*100))
    if len(v) < 6:
        v = '0' + v
    return prefix + v + '.jpg'

def cv2sketch(img, args):
    for sigma_s_val in range(5, 35, 5):
        for sigma_r_val in map(lambda x: x/100.0, range(10, 15, 5)):
            for shade_factor_val in map(lambda x: x/100.0, range(10, 15, 5)):
                sketch_gray, sketch_color = cv2.pencilSketch(img, sigma_s=sigma_s_val, sigma_r=sigma_r_val, shade_factor=shade_factor_val)
                file_name = get_file_name(args.file.replace('.jpg', '') + '/sketch_', sigma_s_val, sigma_r_val, shade_factor_val)
                cv2.imwrite(file_name, sketch_gray)
                print('written to ' + file_name)

def stylize(img, args):
    for sigma_s_val in range(5, 15, 5):
        for sigma_r_val in map(lambda x: x/100.0, range(10, 15, 5)):
            stylize = cv2.stylization(img, sigma_s=sigma_s_val, sigma_r=sigma_r_val)
            file_name = get_file_name(args.file.replace('.jpg', '') + '/stylize_', sigma_s_val, sigma_r_val)
            cv2.imwrite(file_name, stylize)
            print('written to ' + file_name)

def dodge(front, back):
    result=front*255/(255-back) 
    result[np.logical_or(result > 255, back == 255)] = 255
    return result#.astype('uint8')

def denoise(args, suffix="_greyscale", smoothing = 280):
    img = cv2.imread(args.file.replace('.jpg', '') + suffix + ".jpg")
    img = 255 * (img > 128).astype(np.uint8)

    # Create the graph.
    g = maxflow.Graph[int]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(img.shape)
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, smoothing)
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    g.add_grid_tedges(nodeids, img, 255-img)

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    img_denoised = np.logical_not(sgm).astype(np.uint8) * 255
    return img_denoised

def sharpen(img):
    blur_img2 = scipy.ndimage.filters.gaussian_filter(img, 3)
    filter_blurred2 = scipy.ndimage.gaussian_filter(blur_img2, 1)
    alpha = 30
    sharpened = blur_img2 + alpha * (blur_img2 - filter_blurred2)
    return sharpened

def canny(img, args):
    edges = cv2.Canny(img,110,180)
    edges_inv_img = np.invert(edges)
    cv2.imwrite(args.file.replace(".jpg", "_canny.jpg"), edges_inv_img.astype('uint8'))

def new_sketch(img, args):
    gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    gray_inv_img = 255-np.dot(img[...,:3], [0.299, 0.587, 0.114])
    blur_img = scipy.ndimage.filters.gaussian_filter(gray_inv_img, 5)
    first_pencil_img = dodge(blur_img, gray_img)
    cv2.imwrite(args.file.replace(".jpg", "_sketch.jpg"), first_pencil_img.astype('uint8'))
    cv2.imwrite(args.file.replace(".jpg", "_sketch_sharpened.jpg"), sharpen(first_pencil_img).astype('uint8'))
    cv2.imwrite(args.file.replace(".jpg", "_sketch_denoised.jpg"), denoise(args, "_sketch_sharpened").astype('uint8'))
    os.unlink(args.file.replace(".jpg", "_sketch.jpg"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.getcwd())
    parser.add_argument('--file', default='x.jpg')
    args = parser.parse_args()
    if not os.path.exists(args.file.replace('.jpg', '')):
        os.makedirs(args.file.replace('.jpg', ''))
    cv2sketch(cv2.imread(args.file), args)
    stylize(cv2.imread(args.file), args)
    new_sketch(cv2.imread(args.file), args)
    canny(cv2.imread(args.file), args)

if __name__ == '__main__':
    main()
