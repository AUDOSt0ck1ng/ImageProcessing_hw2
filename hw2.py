import os
import cv2
import numpy as np
import glob
from scipy.fftpack import fft

#app_win_size_x = 870
#app_win_size_y = 500

output_dir_path = output_hi_path = output_um_path = output_lo_path = ""

@staticmethod
def root_path(): #當前 working dir 之 root path
    return os.getcwd()
    #return "/workspaces/mvl/ImageProcessing"

def set_output_path():
    global output_dir_path, output_lo_path, output_um_path, output_hi_path
    output_dir_path = os.path.join(root_path(), "hw2_output")
    output_lo_path = os.path.join(output_dir_path, "Laplacian_operator")
    output_um_path = os.path.join(output_dir_path, "unsharp_masking")
    output_hi_path = os.path.join(output_dir_path, "high_boost")
    
def get_output_path():
    global output_dir_path, output_lo_path, output_um_path, output_hi_path
    return [output_dir_path, output_lo_path, output_um_path, output_hi_path]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("mkdir "+dir_path)    
    else:
        print(dir_path+" already exist, no need to mkdir.")

def get_image_path(path): #root_path/HW2_test_image
    return glob.glob(os.path.join(path, "*.bmp"))+glob.glob(os.path.join(path, "*.tif"))

def show_img_fullscreen(img_name, showimg ,type):
    cv2.namedWindow(img_name, type)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(img_name, app_win_size_x,app_win_size_y)
    #cv2.moveWindow(img_name, app_pos_x,app_pos_y)
    cv2.imshow(img_name, showimg)

def read_and_operate_image(image_path):
    image =cv2.imread(image_path)
    #show_img_fullscreen("Current Image: "+image_path, image, cv2.WINDOW_KEEPRATIO)
    image_gray =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #show_img_fullscreen("Current Image(grayscale): "+image_path, image_gray, cv2.WINDOW_KEEPRATIO)
    return image, image_gray

#laplacian_operator
"""-------------------------------
padding先把矩陣補成(m+p*2)x(n+p*2)，convolution之後會變回mxn。
p = LM 的 kernel size/2
ex:
Laplacian matrix 3x3 (kernel size 3)
LM = [[0, 1, 0],[1, -4, 1], [0, 1, 0]]
p = 1
----------------------------------
formula:
new_point = c·sum(LM convolution img) + point

計算後的值可能會超出範圍、以上下值取代。
-------------------------------"""
def laplacian_operator(img):
    M, N = img.shape
    k=3 # kernel
    c = -1
    p = k // 2
    new_img = np.zeros((M + p * 2, N + p * 2), dtype=np.float64)
    new_img[p: p + M, p: p + N] = img.copy().astype(np.float64)
    tmp = new_img.copy()

    # laplacian matrix
    LM = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]
    for m in range(M):
        for n in range(N):
            point = tmp[p + m, p + n]
            con = tmp[m: m + k, n: n + k]
            new_img[p + m, p + n] = c*np.sum(LM*con) + point
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img[p: p + M, p: p + N].astype('uint8')

    F_img = np.fft.fft2(img)
    F_filter = np.fft.fft2(np.array([[0,-1,0], [-1,4,-1], [0,-1,0]]), s=(img.shape))
    F_H_img = F_img+F_img*F_filter
    new_img_fft = np.abs(np.fft.ifft2(F_H_img))

    return new_img, new_img_fft
    
# unsharp masking
def unsharp_masking_s(img):
    input = img.copy().astype(np.int32)
    img_blur = cv2.GaussianBlur(img, (31, 31), 0, 0 ) #卷積
    sharp = input - img_blur
    k = 1
    sharp = input + k*sharp
    sharp = np.uint8(np.clip(sharp,0,255))
    return sharp

def unsharp_masking_f(img):
    f= img
    F = np.fft.fftshift(np.fft.fft2(f))
    M,N = F.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 15
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0*D0))

    FLP = H * F

    FLP = np.fft.ifftshift(FLP)
    fLP = np.abs(np.fft.ifft2(FLP))

    sharp = f - fLP

    # unsharp masking
    k = 1
    g = f + k*sharp

    g = np.clip(g, 0, 255)
    return g

# high boost filter
def high_boost_s(img):
    input = img.copy().astype(np.int32)
    img_blur = cv2.GaussianBlur(img, (31, 31), 0, 0 ) #卷積
    sharp = input - img_blur
    A = 2.5
    sharp = (A-1)*input + sharp
    sharp = np.uint8(np.clip(sharp,0,255))
    return sharp

def high_boost_f(img):
    M, N = img.shape
    A = 2.5
    D0 = 15
    #image and filter FFT
    f = np.fft.fftshift(np.fft.fft2(img))
    new_image = f.copy()
    #calculate filter in frequency domain
    for u in range(0, M):
        for v in range(0, N):
            D = (((u - M/2)**2) + ((v - N/2)**2))**0.5
            new_image[u, v] = (A-1) * f[u, v] + f[u, v] - f[u, v] * np.exp(-(D**2)/(2*(D0**2)))
    new_image = np.fft.ifftshift(new_image)
    #image iFFT
    new_image = np.fft.ifft2(new_image)
    new_image = np.real(new_image)
    new_image= np.clip(new_image, 0, 255)
    return new_image
# main
image_dir = os.path.join(root_path(), "HW2_test_image")
print(image_dir)
images = get_image_path(image_dir)  #取得圖片路徑
print(images)

set_output_path()
for output_path in get_output_path():
    mkdir(output_path)

dir, lo, um, hi = get_output_path()

for image in images:
    file = image.replace(".bmp", "").replace(".tif", "")
    img, img_gray = read_and_operate_image(image)
    img_lo_s, img_lo_f = laplacian_operator(img_gray)
    img_um_s = unsharp_masking_s(img_gray)
    img_um_f = unsharp_masking_f(img_gray)
    img_hi_s = high_boost_s(img_gray)
    img_hi_f = high_boost_f(img_gray)

    cv2.imwrite(file.replace(image_dir, lo)+".bmp", img)
    cv2.imwrite(file.replace(image_dir, lo)+"_laplacian_operator_s.bmp", img_lo_s)
    cv2.imwrite(file.replace(image_dir, lo)+"_laplacian_operator_f.bmp", img_lo_f)
    cv2.imwrite(file.replace(image_dir, um)+".bmp", img)
    cv2.imwrite(file.replace(image_dir, um)+"_unsharp_masking_s.bmp", img_um_s)
    cv2.imwrite(file.replace(image_dir, um)+"_unsharp_masking_f.bmp", img_um_f)
    cv2.imwrite(file.replace(image_dir, hi)+".bmp", img)
    cv2.imwrite(file.replace(image_dir, hi)+"_high_boost_s.bmp", img_hi_s)
    cv2.imwrite(file.replace(image_dir, hi)+"_high_boost_f.bmp", img_hi_f)