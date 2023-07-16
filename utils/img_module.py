import os
import cv2
import glob
import numpy as np

class ImageModule():
    
    def __init__(self):
        pass
    
    def read_image(self, filepath):
        '''
            filepath : 이미지 저장된 경로 (이미지 저장된 상위 폴더의 경로)
        '''
        # files = glob.glob(filepath+'*.png')
        print("filepath:", filepath)
        images = []
        names = []
        # for f in files:
        image = cv2.imread(filepath)
        images.append(image)
        # names.append(os.path.splitext(f)[0].split('\\')[-1])
        
        return images
    
    ##? 이미지 리스트 받아와 resize하는 함수 
    def resize_image(self, images, width:int, height:int):
        
        re_images = []   ##resized images
        for img in images:
            re_image = cv2.resize(img, (width, height))
            re_images.append(re_image)
        
        return re_images
    
    #이미지를 특정 폴더에 저장
    def save_images(self, images:list, dest, names=None):
        '''
            name = None  설정시 번호 넘버링하여 저장
                    name  이름 그대로 저장
        '''
        for idx, img in enumerate(images):
            if names == None:
                cv2.imwrite('{}{}.png'.format(dest, idx), img)
                print("{}번째 이미지 저장.".format(idx))
            else:
                cv2.imwrite('{}{}.png'.format(dest, names[idx]), img)
                print("{}번째 이미지 저장----{}.".format(idx, names[idx]))
                
    
    '''
    #이미지 색상을 추출한다
    def extract_color(self, images:list):
        for idx, img in enumerate(images):
            print(img)
            print(dir(img))
            cv2.imshow("test",img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    '''
    
    #이미지 색상을 반전시킨다
    def reverse_bg(self, image) :
        '''
        images : cv2.imread한 이미지들의 리스트 
        '''
        result = []
        
        # for idx, img in enumerate(images):
        encoded_img = np.frombuffer(image.numpy(), dtype=np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        # 그레이스케일로 변환
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 반전시키기
        inverted_img = cv2.bitwise_not(gray_img)
            # cv2.imshow('inverted Image', inverted_img)
            # cv2.waitKey(0)
        ret, thresh = cv2.threshold(inverted_img, 150, 255, cv2.THRESH_BINARY)
            
        # 이미지 보여주기
        # cv2.imshow('thresh Image', thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # 컨투어 찾기
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 컨투어 그리기
        contour_img = cv2.drawContours(inverted_img, contours, -1, (255, 255, 0), 1)
    
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print('Contour area:', area)
            if area <= 100:
                # 컨투어 내부 영역의 색상을 255로 변경
                cv2.drawContours(inverted_img, [cnt], 0, 200, -1)
                

            # 변경된 이미지 보여주기
            # cv2.imshow('Image', inverted_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        result.append(inverted_img)
        return result[0]
       
        