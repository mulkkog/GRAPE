import os
from PIL import Image

def calculate_average_image_size(directory):
    widths = []
    heights = []

    # 디렉토리 안의 모든 파일 확인
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)

    if len(widths) == 0:
        print("이미지가 없습니다.")
        return None

    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)
    print(f"이미지 개수: {len(widths)}")
    print(f"평균 크기: {avg_width:.2f} x {avg_height:.2f}")

    return avg_width, avg_height

# 사용 예시
directory = 'data/1_Image_SR/DIV2K/DIV2K_valid_HR'
calculate_average_image_size(directory)
