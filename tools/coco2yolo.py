import json
import os
import shutil

# COCO Annotation json 파일 YOLO Annotation으로 수정

# COCO JSON 파일 경로
train_coco_annotation_path = '/your_train_annotation_path'
val_coco_annotation_path = '/your_val_annotation_path'

# YOLO 형식의 텍스트 파일이 저장될 디렉토리
train_output_dir = '/train_save_ann_directory'
val_output_dir = '/val_save_ann_directory'

# 실제 이미지가 저장될 디렉토리
train_image_output_dir = '/train_save_img_directory'
val_image_output_dir = '/val_save_img_directory'

# 원본 이미지 경로 (여기서 각 이미지의 상대 경로를 읽어서 처리)
image_src_dir = '/your_image_path'

# 디렉토리 생성 (이미 존재하면 에러 없음)
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(train_image_output_dir, exist_ok=True)
os.makedirs(val_image_output_dir, exist_ok=True)

# COCO JSON 파일을 YOLO 형식으로 변환하고 이미지를 학습/검증용으로 분리하는 함수
def convert_and_copy_images(coco_annotation_path, output_dir, image_output_dir, image_src_dir):
    # 주석 파일을 불러오기
    with open(coco_annotation_path) as f:
        coco_data = json.load(f)
    
    # 이미지 크기 정보와 파일 이름을 저장
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 주석을 YOLO 형식으로 변환하여 저장
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']  # YOLO는 클래스 ID가 0부터 시작
        bbox = annotation['bbox']
        
        # 이미지의 너비와 높이
        image_width, image_height = image_id_to_size[image_id]
        
        # COCO 형식에서 YOLO 형식으로 변환
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width /= image_width
        height /= image_height
        
        # YOLO 형식의 라인
        yolo_annotation = f"{category_id} {x_center} {y_center} {width} {height}\n"
        
        # YOLO 형식의 텍스트 파일 저장 경로
        output_file = os.path.join(output_dir, f"{str(image_id).zfill(4)}.txt")
        
        # YOLO 텍스트 파일에 쓰기 ('a' 모드로 설정하여 덮어쓰기)
        with open(output_file, 'a') as file:
            file.write(yolo_annotation)
        
        # 이미지 파일 복사
        image_filename = os.path.basename(image_id_to_filename[image_id])  # 이미지 파일명만 추출
        src_image_path = os.path.join(image_src_dir, image_filename)  # 원본 경로에 파일명만 추가
        dest_image_path = os.path.join(image_output_dir, image_filename)  # 타겟 디렉토리에는 파일명만 복사
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dest_image_path)
        else:
            print(f"이미지 {src_image_path}를 찾을 수 없습니다.")

# 학습용 및 검증용 COCO 주석 파일을 변환하고 이미지 분리
convert_and_copy_images(train_coco_annotation_path, train_output_dir, train_image_output_dir, image_src_dir)
convert_and_copy_images(val_coco_annotation_path, val_output_dir, val_image_output_dir, image_src_dir)

print(f"YOLO 형식으로 변환된 텍스트 파일이 {train_output_dir}, {val_output_dir}에 저장되었습니다.")
print(f"이미지 파일이 {train_image_output_dir}, {val_image_output_dir}로 복사되었습니다.")
