import lmdb, os, io, struct
from pathlib import Path
from PIL import Image
import numpy as np

lmdb_path   = "/home/jijang/ssd_data/projects/ContinuousSR/data/3_Real_Image_Denoising/SIDD/val/gt_crops.lmdb"          # data.mdb 있는 폴더
out_dir     = "/home/jijang/ssd_data/projects/ContinuousSR/data/3_Real_Image_Denoising/SIDD/val/gt_crops"          # 복구한 이미지 저장 폴더
decode_type = "png"                   # "png" 또는 "raw"

Path(out_dir).mkdir(exist_ok=True)

with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            key = k.decode() if isinstance(k, bytes) else str(k)

            # ---------- ① 값이 PNG 바이트인 경우 ----------
            if decode_type == "png":
                img = Image.open(io.BytesIO(v)).convert("RGB")
                img.save(f"{out_dir}/{key}.png")

            # ---------- ② 값이 raw(uint8) 배열인 경우 ----------
            # meta 예: “ValidationBlocksSrgb_102.png (256,256,3) 1”
            # 이름·shape를 키로부터 추출했다고 가정
            if decode_type == "raw":
                # (256,256,3) → [256,256,3]
                shape = tuple(map(int, key.split("(")[1].split(")")[0].split(",")))
                arr   = np.frombuffer(v, dtype=np.uint8).reshape(shape)
                img   = Image.fromarray(arr[:, :, :3], mode="RGB")  # 알파 등 필요시 수정
                img.save(f"{out_dir}/{key.split()[0]}.png")
