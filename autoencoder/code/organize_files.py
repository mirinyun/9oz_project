import os
import shutil

data_path_train = './content/orig_A20+21+23'  # 트레이닝 이미지가 있는 폴더 경로
output_path_train = './content/orig_sorted_A20+21+23'  # 클래스별로 정리된 폴더가 생성될 경로

# 클래스별로 폴더를 생성하고 이미지를 해당 폴더로 이동하는 함수
def organize_images(data_path, output_path):
    os.makedirs(output_path, exist_ok=True)  # 출력 폴더 생성

    for filename in os.listdir(data_path):
        if filename.endswith('.jpg'):  # 확장자가 png인 경우에만 처리
            class_name = filename[4:6]
            class_path = os.path.join(output_path, class_name)

            os.makedirs(class_path, exist_ok=True)  # 클래스별 폴더 생성

            old_path = os.path.join(data_path, filename)
            new_path = os.path.join(class_path, filename)

            shutil.move(old_path, new_path)  # 이미지 이동

# 이미지 정리 수행
organize_images(data_path_train, output_path_train)