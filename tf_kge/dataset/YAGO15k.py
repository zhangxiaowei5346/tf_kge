# coding=utf-8
from tf_kge.dataset.common import CommonDataset


class YAGO15kDataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="YAGO15k",
                         download_urls=[
                             "https://github.com/zhangxiaowei5346/TKDE_DATA/raw/main/YAGO15k.zip"
                         ],
                         download_file_name="YAGO15k.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)

